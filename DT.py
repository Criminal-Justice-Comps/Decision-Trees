'''decisiontree.py written Nov 2019

A program for creating a decision tree on a binary classification problem.

TO RUN: use the command line as follows:
    python3 DT.py _filename_ (optional arguments)
Where "_filename_" is the path to the data that should be trained on.
The optional arguments are as follows:
    --hideTree Turns off the display of the trees.
    --saveTree sets the filepath to save the tree in. Default value is 'save_tree.txt'.
        If you do not want to save the output, use "--saveTree NULL" (caps required).
    --savePred sets the filepath to save the predictions and correct labels in.
        By default, "save_pred.txt" is used. If you do not want to save the output,
        use "--savePred NULL" (caps required).
    --trainCount, the number of data points to use as the training set.

When run, this program will either train or load a decision tree, then test the tree
to check accuracy.

In this program, we assume that:
    The data is stored in a comma separated text file.
    Ground truth is stored in the last column of the file.
    First row of the file contains feature names.
    First column of the file contains IDs, and should not be looked at as a feature.


'''

from collections import defaultdict, Counter
from time import time
import math
import copy
import argparse
import numpy as np
import json

# column numbers - 1, with the first column being column 0 (and also not being looked at)
NUMERIC_FEATURES = [2, 3, 4, 5, 6]
JSON_FILE_PATH = "../Fairness/DecisionTreesData.json"

def parse_args():
    ''' Parse command line arguments '''
    p = argparse.ArgumentParser()
    p.add_argument("filepath", default="tennis.txt", help="Path to text file containing training data.")
    p.add_argument("--hideTree", action="store_true", help="Turn off the display of all trees.")
    p.add_argument("--saveTree", default='save_tree.txt', help='Filepath to save the tree in.')
    p.add_argument("--savePred", default='save_pred.txt', help='Filepath to save the predictions in.')
    p.add_argument("--trainCount", type=int, default=100, help='Number of data points to use as the training set.')
    p.add_argument("--maxDepth", type=int, default=-1, help="The maximum depth to build the DT to. By default, no maximum depth is set.")
    args = p.parse_args()
    return args

class Node:
    ''' A node class for the decision tree.
            Variables are as follows:
                parent: the parend of the node
                attribute: what the node splits over. Always None if the node is a leaf.
                value: the value of the node's atribute that the node will split on
                children: the children of the node, stored in a dictionary where the keys
                          are all possible values of the node's attribute variable.
                          Always None if the node is a leaf.
                guess: the class the node will assign ito its examples.
                       Always None if the node is not a leaf.

            Methods are as follows:
                prune(guess): makes the node a leaf with node.guess = parameter guess
                set_guess: set the guess of the node to the proper value based on the node's data
                classify(point): returns the value that the node (or its children)
                                 assign to the point passed
                get_highest_count(labels): returns the class with the highest representation
                                           within the labels passed.
                '''

    def __init__(self, parent, value=None, attribute=None, leaf_class=None):
        self.parent = parent
        if leaf_class == None:
            self.attribute = attribute
            self.children = {}
            self.value = value
            self.guess = None
        else:
            self.attribute = None
            self.children = None
            self.value = None
            self.guess = leaf_class

    def prune(self, guess):
        '''Makes the node a leaf with node.guess = parameter guess'''
        self.attribute = None
        self.children = None
        self.value = None
        self.guess = guess

    def set_guess(self, labels, default):
        self.attribute = None
        self.value = None
        self.children = None
        if len(labels) == 0:
            self.guess = default
        else:
            self.guess = self.get_highest_count(labels)

    def classify(self, point):
        '''Returns the value that the node (or its children) assign to the point passed '''
        if self.guess != None:
            return self.guess
        else:
            if self.attribute in NUMERIC_FEATURES:
                if point[self.attribute] > self.value:
                    return self.children[1].classify(point)
                else:
                    return self.children[0].classify(point)
            else:
                return self.children[point[self.attribute]].classify(point)

    def get_highest_count(self, labels):
        count = Counter(labels)
        return count.most_common(1)[0][0]



def make_tree(node, examples, labels, attributes, features_with_values, default_guess=0, max_depth = math.inf, depth = 0):
    '''Creates a decision tree from the current node down, with max depth = len(attributes) if not set by user'''
    if len(attributes) > 0:
        depth += 1
        curr_entropy = entropy(labels)
        exp_entropy, best_choice, best_value = entropy_exp(examples, labels, attributes)

        if curr_entropy <= exp_entropy or depth > max_depth:
            # cannot decrease entropy, so terminate this branch
            node.set_guess(labels, default_guess)
            return 0

        node.attribute = best_choice
        attributes.remove(best_choice)

        examples_dict, labels_dict = split_data(examples, labels, best_choice, best_value, features_with_values)
        default_guess = node.get_highest_count(labels)

        if best_choice in NUMERIC_FEATURES:
            node.value = best_value
            node.children = {0: Node(node), 1: Node(node)}
            make_tree(node.children[0], examples_dict[0], labels_dict[0], copy.copy(attributes), features_with_values, default_guess, max_depth, depth)
            make_tree(node.children[1], examples_dict[1], labels_dict[1], copy.copy(attributes), features_with_values, default_guess, max_depth, depth)
        else:
            for value in features_with_values[best_choice]:
                node.children[value] = Node(node)
                make_tree(node.children[value], examples_dict[value], labels_dict[value], copy.copy(attributes), features_with_values, default_guess, max_depth, depth)

    else:
        # current node is a leaf, so terminate this branch
        node.set_guess(labels, default_guess)
        return 1



def entropy_exp(examples, labels, attributes):
    ''' Returns the expected entropy and choice of attribute which minimizes entropy'''
    if len(examples) == 0:
        # here we return math.inf because for the entropy because that will force the
        # current node to be a leaf, as it should be with no examples found.
        return math.inf, attributes[0], 0
    poss_index = 0 # keep track of the index of the best attribute
    best_choice = attributes[poss_index]
    best_entropy = math.inf
    best_value = 0
    while poss_index < len(attributes):

        if attributes[poss_index] in NUMERIC_FEATURES:
            curr_entropy, value = get_best_split_value(examples, labels,
                                                            attributes[poss_index])
        else:
            curr_entropy, value = get_categorical_entropy(examples, labels,
                                                               attributes[poss_index])

        if (curr_entropy) < best_entropy:
            best_choice = attributes[poss_index]
            best_entropy = curr_entropy
            best_value = value
        poss_index += 1
    #print(best_entropy, best_choice, "**")
    return best_entropy, best_choice, best_value

def get_categorical_entropy(examples, labels, attribute):
    split_examples = defaultdict(list)
    labeled = [[examples[i], labels[i]] for i in range(len(examples))]
    for example in labeled:
        split_examples[example[0][attribute]].append(example)
    entr = 0
    for key in split_examples:
        split_labels = [example[1] for example in split_examples[key]]
        entr += (len(split_labels)/len(labeled)) * entropy(split_labels)
    return entr, 0



def get_best_split_value(examples, labels, attribute):
    ''' For a given attribute, calculates the value of that attribute which minimizes
    entropy when the given data is split over that value of the given attribute.'''
    # first sort the data and labels so that we can change index rather than value
    # to avoid checking values which do not change the number of examples on either
    # side of the best split.
    labeled = [[examples[i], labels[i]] for i in range(len(examples))]
    in_order = sorted(labeled, key=lambda example: example[0][attribute])
    sorted_labels = [example[1] for example in in_order]
    # calculate the index to split on
    best_entropy = math.inf
    best_num = 0 # stores the index corresponding to the best value to split over
    curent_value = None
    # in this loop, i corresponds to the number of items below the split
    for i in range(len(sorted_labels)):
        # there is a small problem calculating entropy when there are no examples
        # to calculate from--because we are using weighted entropy we let that
        # value = 0 for the sake of practicality.
        below_entr = 0
        if i != 0:
            below_entr = i/len(sorted_labels) * entropy(sorted_labels[:i])
        above_entr = 0
        if i != len(sorted_labels):
            above_entr = (len(sorted_labels) - i)/len(sorted_labels) * entropy(sorted_labels[i:])
        if (above_entr + below_entr) <= best_entropy:
            # here we use less than or equal to to get the highest index corresponding
            #      to the desired value of the given attribute
            best_entropy = above_entr
            best_num = i
    best_value = in_order[best_num - 1][0][attribute] # get the value to split the data on
    return best_entropy, best_value


def entropy(labels):
    '''Returns the entropy of the given set of labels. Here we assume that the
    entropy of a class without examples in the labels is 0.'''
    entr = 0
    count = Counter(labels)
    for key in count:
        count[key] = count[key]/len(labels)
        entr += count[key] * math.log2(1/count[key])
    return entr


def split_data(examples, labels, attribute, value, features_with_values):
    '''Split bivariate categorical data and associated labels according to attribute
        parameter.'''
    examples_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    if attribute in NUMERIC_FEATURES:
        for i in range(len(examples)):
                if int(examples[i][attribute]) > int(value):
                    examples_dict[1].append(examples[i])
                    labels_dict[1].append(labels[i])
                else:
                    examples_dict[0].append(examples[i])
                    labels_dict[0].append(labels[i])
    else:
        for i in range(len(examples)):
            examples_dict[examples[i][attribute]].append(examples[i])
            labels_dict[examples[i][attribute]].append(labels[i])
    return examples_dict, labels_dict


def get_data(txt_file_name, train_test_split):
    '''Returns a list of all data points and all data labels, split into training
        and testing sets.'''
    with open(txt_file_name) as file:
        features = []
        features_with_values = defaultdict(list)
        training = []
        training_labels = []
        testing = []
        testing_labels = []
        i = 0
        for line in file:
            split_line = line.split(",")
            split_line[-1] = split_line[-1].rstrip('\n')
            if i == 0:
                features = split_line[1:]
            elif i < train_test_split:
                person = [el for el in split_line[1:-1]]
                training.append(person)
                training_labels.append(int(split_line[-1]))
                # add any new values to features_with_values
                for j in range(len(person)):
                    if person[j] not in features_with_values[j]:
                        features_with_values[j].append(person[j])
            elif i > train_test_split:
                person = [el for el in split_line[1:-1]]
                testing.append(person)
                testing_labels.append(int(split_line[-1]))
                # add any new values to features_with_values
                for k in range(len(person)):
                    if person[k] not in features_with_values[features[k]]:
                        features_with_values[features[k]].append(person[k])
            i += 1
    return training, training_labels, testing, testing_labels, features, features_with_values


def save_tree(filepath, features, root):
    with open(filepath, 'w') as file:
        string = get_tree_string(root, features)
        file.write(string)
    return 1

def get_tree_string(node, features, depth = 0, string=''):
    ''' Returns a string representation of a DT to be saved for later.
    The tree is saved in the same format that it is printed in.
    '''
    if node.guess == None:
        string += "\n"
        for value, child in node.children.items():
            if depth != 0:
                string += "|\t"*(depth-1)
                string += "|\t"*depth
            if node.attribute in NUMERIC_FEATURES:
                if value == 1:
                    string += str(features[node.attribute])
                    string +=  ">"
                    string += str(node.value)
                else:
                    string += str(features[node.attribute])
                    string += "<="
                    string += str(node.value)
            else:
                string += str(features[node.attribute])
                string += str(value)
            string = get_tree_string(child, features, depth+1, string)
    else:
        string += ": "
        string += str(node.guess)
        string += "\n"
    return string


def display_tree(node, features, depth=0):
    '''Prints out the tree with root 'node'.'''
    if node.guess == None:
        print()
        for value, child in node.children.items():
            if depth != 0:
                print("|\t"*(depth-1), end='')
                print("|\t"*depth, end='')
            if node.attribute in NUMERIC_FEATURES:
                if value == 1:
                    print(features[node.attribute], ">", node.value, end='')
                else:
                    print(features[node.attribute], "<=", node.value, end='')
            else:
                print(features[node.attribute], value, end="")
            display_tree(child, features, depth+1)
    else:
        print(": ", end='')
        print(node.guess)


def main():
    '''Creates and prints a decision tree on a binary classification problem.'''
    args = parse_args()
    txt_file_name = args.filepath
    root = Node(None)
    end_time = time()
    start_time = time()
    training, training_labels, testing, testing_labels, features, features_with_values = get_data(txt_file_name, args.trainCount)
    attributes = list(np.arange(len(training[0])))

    if args.maxDepth <= 0:
        make_tree(root, training, training_labels, attributes, features_with_values)
    else:
        make_tree(root, training, training_labels, attributes, features_with_values, 0, args.maxDepth)

    end_time = time()
    if not args.hideTree:
        display_tree(root, features)
        print()
    num_correct = 0
    predictions = [] # a list of recidivism guesses (0s and 1s)
    for i in range(len(testing)):
        test = testing[i]
        label = testing_labels[i]
        pred = root.classify(test)
        test.append(pred)
        test.append(label)
        predictions.append(test)
        if pred == int(label):
            num_correct += 1

    if len(testing) == 0:
        print("No testing data provided.")
        print("len(training)", len(training))
    else:
        print("Tree has accuracy:", num_correct/len(testing))

    print("Training time was", end_time - start_time)

    if args.saveTree != 'NULL':
        save_tree(args.saveTree, features, root)

    if args.savePred != 'NULL':
        string = 'sex,race,age,juv_fel_count,juv_misd_count,juv_other_count,priors_count,prediction,truth\n'
        for person in predictions:
            for el in person:
                string += str(el)
                string += ','
            string = string[:-1]
            string += '\n'
        string = string[:-1]
        with open(args.savePred, 'w') as file:
            file.write(string)








if __name__ == "__main__":
    main()
