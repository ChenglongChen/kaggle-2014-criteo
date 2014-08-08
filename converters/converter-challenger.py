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

NR_BINS = args['nr_bins']

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = set()

        for j in range(1, 14):
            if j in [1]:
                continue
            value = row['I{0}'.format(j)]
            if j == 5 and value != '':
                value = int(math.log(float(value)+1))
            elif j in [2, 3, 6, 7, 9] and value != '':
                value = int(float(value)/10)
            bin = hashstr(str(j)+str(value), NR_BINS)
            feats.add((bin, 0.16))

        for j in range(1, 27):
            if j in [11, 21]:
                continue
            value = row['C{0}'.format(j)]
            if value == '':
                bin = hashstr(str(j)+str(value), NR_BINS)
            else:
                bin = int(value, 32)%NR_BINS+1
            feats.add((bin, 0.16))

        feats = list(feats)
        feats.sort()
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
