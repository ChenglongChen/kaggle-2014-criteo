#!/usr/bin/env python3

import argparse, csv, sys, math, itertools

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='convert to svm')
parser.add_argument('-n', default=10000000, type=int, help='set number of bins for hashing trick')
parser.add_argument('-c', default=0, type=int, help='set number of bins for hashing trick')
parser.add_argument('-i', default=0, type=int, help='set number of bins for hashing trick')
parser.add_argument('csv_path', type=str, help='set path to the csv file')
parser.add_argument('svm_path', type=str, help='set path to the svm file')
args = vars(parser.parse_args())

bins = {1: 57, 2: 2576, 3: 655, 4: 9, 5: 231594, 6: 4310, 7: 563, 8: 60, 9: 290, 10: 1, 11: 1, 12: 40, 13: 73}

with open(args['svm_path'], 'w') as f:
    for i, row in enumerate(csv.reader(open(args['csv_path']))):
        if i == 0:
            continue
        feats = set()
        label = row[1]
        for i, element in enumerate(row[2:15], start=1):
            if i not in [1, 4, 8, 10, 11, 12, 13]:
                continue
            bin = hashstr(str(i)+str(element), args['n'])
            feats.add((bin, 1))

        for i, element in enumerate(row[15:], start=1):
            if i not in [2, 5, 6, 8, 9, 14, 17, 20, 22, 23, 25]:
                continue
            bin = hashstr(str(i+20)+element, args['n'])
            feats.add((bin, 1))

        feats = list(feats)
        feats.sort()
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(label + ' ' + ' '.join(feats) + '\n')
