#!/usr/bin/env python3

import argparse, csv, sys, itertools, collections

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

feat_pool = set()

def gen_categorical_feats(row):
    feats = []
    for j in range(1, 27):
        field = 'C{0}'.format(j)
        value = row[field]
        key = field + '-' + value
        feats.append(key)
    return feats

for i, row in enumerate(csv.DictReader(open(args['csv_path'])), start=1):
    feats = gen_categorical_feats(row)
    for feat in feats:
        feat_pool.add(feat)
    if i % 100000 == 0:
        sys.stderr.write('{0:7}k {1:7}k\n'.format(int(i/1000), int(len(feat_pool)/1000)))
sys.stderr.write('nr_feats = {0}\n'.format(len(feat_pool)))

with open(args['out_path'], 'w') as f:
    for feat in feat_pool:
        f.write('{0}\n'.format(feat))
