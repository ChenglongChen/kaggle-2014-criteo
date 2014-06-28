#!/usr/bin/env python3

import argparse, csv, hashlib, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='process some integers')
parser.add_argument('-n', action='store', default=1000000, help='set number of bins for hashing trick', type=int)
parser.add_argument('csv_path', type=str, help='set path to the csv file')
parser.add_argument('svm_path', type=str, help='set path to the svm file')
args = parser.parse_args()

NR_BINS, CSV_PATH, SVM_PATH = args.n, args.csv_path, args.svm_path

def hashstr(str):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%NR_BINS

with open(SVM_PATH, 'w') as f:
    for i, row in enumerate(csv.reader(open(CSV_PATH))):
        if i == 0:
            continue
        feats = set()
        for i, element in enumerate(row):
            if i == 0 or element == '':
                continue
            elif i == 1:
                label = element 
            elif i <= 14:
                feats.add((i-1, element))
            else:
                bin = hashstr(str(i)+element)+14
                feats.add((bin, 1))
        feats = list(feats)
        feats.sort()
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(label + ' ' + ' '.join(feats) + '\n')
