#!/usr/bin/env python3

import argparse, csv, hashlib, sys

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
            value = row['I{0}'.format(j)]
            bin = hashstr(str(j)+str(value), NR_BINS)+1
            feats.add((bin, 1))
        for j in range(1, 27):
            value = row['C{0}'.format(j)]
            if value == '':
                bin = hashstr(str(j)+str(value), NR_BINS)+1
            else:
                bin = int(value, 32)%NR_BINS+1
            feats.add((bin, 1))
        feats = list(feats)
        feats.sort()
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
