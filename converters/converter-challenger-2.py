#!/usr/bin/env python3

import argparse, csv, hashlib, sys, math, itertools

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
            feats.add(bin)

        poly2 = []
        for j in range(1, 27):
            if j in [11, 21]:
                continue
            value = row['C{0}'.format(j)]
            if value == '':
                continue
            else:
                poly2.append(value)

        for val1, val2 in itertools.combinations(poly2, 2):
            bin = hashstr(val1+val2, NR_BINS) 
            feats.add(bin)

        feats = list(feats)
        feats.sort()
        val = 1/math.sqrt(float(len(feats)))
        feats = ['{0}:{1}'.format(idx, val) for idx in feats]
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
