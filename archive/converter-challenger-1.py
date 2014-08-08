#!/usr/bin/env python3

import argparse, csv, sys, random

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='convert to svm')
parser.add_argument('-n', '--nr_bins', default=10000000, type=int, help='set number of bins for hashing trick')
parser.add_argument('csv_path', type=str, help='set path to the csv file')
parser.add_argument('--prob_path', type=str, help='set path to the prob file')
parser.add_argument('svm_path', type=str, help='set path to the svm file')
args = vars(parser.parse_args())

with open(args['svm_path'], 'w') as f:
    for i, (row, line) in enumerate(zip(open_csv_skip_first_line(args['csv_path']), open(args['prob_path']))):
        feats = set()
        label = row[1]

        prob = float(line.strip())

        if prob > 0.5:
            feats.add((args['nr_bins']+100, 1))
        else:
            feats.add((args['nr_bins']+101, 1))

        for i, element in enumerate(row[2:15], start=1):
            bin = hashstr(str(i)+str(element), args['nr_bins'])+1
            feats.add((bin, 1))
        for i, element in enumerate(row[15:], start=1):
            bin = hashstr(str(i)+element, args['nr_bins'])+1
            feats.add((bin, 1))
        feats = list(feats)
        feats.sort()
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(label + ' ' + ' '.join(feats) + '\n')