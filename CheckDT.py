'''
CheckDT.py

Analyse output of a decision tree, stored in a txt file as follows:

    pred1,pred2,pred3,....,predN
    truth1,truth2,...,truthN

'''

import argparse


def parse_args():
    ''' Parse command line arguments '''
    p = argparse.ArgumentParser()
    p.add_argument("filepath", default="tennis.txt", help="Path to text file containing training data.")
    p.add_argument("--getDist", action="store_true", help="Calculate distances, rather than Accuracy, AUC, ROC.")
    args = p.parse_args()
    return args

def get_data(filepath):
    data = []
    with open(filepath) as file:
        for line in file:
            data.append(line.split(','))
    return data[0], data[1]

def get_distances(predictions, labels):
    distances = []
    print(len(labels), len(predictions))
    for i in range(len(labels)):
        if labels[i] == '1':
            distances.append(10 - int(predictions[i]))
        else:
            distances.append(int(predictions[i]) - 1)
    return distances

def get_accuracy(predictions, labels):
    num_correct = 0
    for i in range(len(predictions)):
        if labels[i] == predictions[i]:
            num_correct += 1
    return num_correct/len(predictions)

def get_roc(predictions, truth):
    return -1

def get_aoc(predictions, truth):
    return -1

def main():
    args = parse_args()
    predictions, truth = get_data(args.filepath)
    if args.getDist:
        distances = get_distances(predictions, truth)
        print("Mean distance is:", sum(distances)/len(distances))
    else:
        print("Accuracy is:", get_accuracy(predictions, truth))



if __name__ == '__main__':
    main()
