#!/usr/bin/env python3

import argparse, csv, hashlib, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='convert to svm')
parser.add_argument('-n', action='store', default=10000000, help='set number of bins for hashing trick', type=int)
parser.add_argument('-i', nargs='+', default=[], help='set numerical fields that are not to be treated as categorical', type=int)
parser.add_argument('--', dest='dummy', action='store_true', help='stop parsing optional arguments')
parser.add_argument('csv_path', type=str, help='set path to the csv file')
parser.add_argument('svm_path', type=str, help='set path to the svm file')
args = vars(parser.parse_args())
args['i'] = set(args['i'])

max_values = {1: 5775.0, 2: 257675.0, 3: 65535.0, 4: 969.0, 5: 23159456.0, 6: 431037.0, 7: 56311.0, 8: 6047.0, 9: 29019.0, 10: 11.0, 11: 231.0, 12: 4008.0, 13: 7393.0}

def hashstr(str):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%args['n']

with open(args['svm_path'], 'w') as f:
    for i, row in enumerate(csv.reader(open(args['csv_path']))):
        if i == 0:
            continue
        feats = set()
        for i, element in enumerate(row):
            if i == 0 or element == '':
                continue
            elif i == 1:
                label = element 
            elif i <= 14:
                key = i-1
                if key != 5:
                    bin = hashstr(str(i)+str(element))+14
                    feats.add((bin, 1))
                else:
                    value = float(element)/max_values[key]
                    if value != 0:
                        feats.add((key, str(round(value, 5))))
            else:
                key = i-14
                if key in set([3, 4, 12, 16, 21]):
                    continue
                bin = hashstr(str(i)+element)+14
                feats.add((bin, 1))
        feats = list(feats)
        feats.sort()
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(label + ' ' + ' '.join(feats) + '\n')
