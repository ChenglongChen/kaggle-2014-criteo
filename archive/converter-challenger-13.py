#!/usr/bin/env python3

import argparse, csv, hashlib, sys, math

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

SKIP_I = set([1])
SKIP_C = set([11, 21])

with open(args['svm_path'], 'w') as f:
    for i, row in enumerate(open_csv_skip_first_line(args['csv_path'])):
        feats = set()
        for i, element in enumerate(row[2:15], start=1):
            if i in SKIP_I:
                continue
            if i == 5 and element != '':
                element = int(math.log(float(element)+1))
            elif i in [2, 3, 6, 7, 9] and element != '':
                element = int(float(element)/10)
            bin = hashstr(str(i)+str(element), args['nr_bins'])
            feats.add((bin, 0.16))

        for i, element in enumerate(row[15:], start=1):
            if i in SKIP_C:
                continue
            if element == '':
                bin = hashstr(str(i+14), args['nr_bins'])
            else:
                bin = int(element, 32)%args['nr_bins']+1
            feats.add((bin, 0.16))

        feats = list(feats)
        feats.sort()
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(row[1] + ' ' + ' '.join(feats) + '\n')