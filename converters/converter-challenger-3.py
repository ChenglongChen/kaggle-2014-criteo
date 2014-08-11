#!/usr/bin/env python3

import argparse, csv, hashlib, sys, math, itertools

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('-t', '--threshold', type=int, default=int(100))
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

NR_BINS = args['nr_bins']

poly2_feat_pool = set()
for row in csv.DictReader(open('stats/fc.trva.r10.p2.t100.txt')):
    poly2_feat_pool.add(row['Key'])

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for feat1, feat2 in itertools.combinations(gen_feats(row), 2):
            feat = feat1 + '-' + feat2
            if feat not in poly2_feat_pool:
                continue
            feats.append(feat)
        feats = gen_hashed_svm_feats(feats, args['nr_bins'])
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
