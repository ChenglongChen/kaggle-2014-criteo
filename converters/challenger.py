#!/usr/bin/env python3

import argparse, csv, hashlib, sys, math

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('-t', '--type', type=str, default='I')
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

if args['type'] != 'I' and args['type'] != 'C':
    print('wrong type')
    exit(1)

frequent_feats = read_freqent_feats(10)

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        raw_feats = gen_feats(row)

        hashed_feats = []
        for feat in raw_feats:
            if feat.startswith('I'):
                if args['type'] != 'I':
                    continue
                hashed_feats.append(feat)
            elif feat.startswith('C'):
                if args['type'] != 'C':
                    continue
                if feat not in frequent_feats:
                    feat = feat[0:2]+'less'
                hashed_feats.append(feat)
            else:
                raise ValueError

        coef = 1/math.sqrt(float(len(raw_feats)))
        hashed_feats = gen_hashed_svm_feats(hashed_feats, args['nr_bins'], coef)
        f.write(row['Label'] + ' ' + ' '.join(hashed_feats) + '\n')
