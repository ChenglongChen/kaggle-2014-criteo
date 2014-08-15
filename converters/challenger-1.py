#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

frequent_feats = read_freqent_feats(args['threshold'])

def gen_hashed_svm_feats_(feats, nr_bins, coef=None):
    feats = [(field, hashstr(feat, nr_bins)) for (field, feat) in feats]
    feats.sort()
    if coef is not None:
        val = coef
    else:
        val = 1/math.sqrt(float(len(feats)))
    feats = ['{0}:{1}:{2}'.format(field, idx, val) for (field, idx) in feats]
    return feats

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for feat in gen_feats(row):
            field = feat.split('-')[0]
            type, field = field[0], int(field[1:])
            if type == 'C' and feat not in frequent_feats:
                feat = feat.split('-')[0]+'less'
            field -= 1
            if type == 'C':
                field += 13
            feats.append((field, feat))
        feats = gen_hashed_svm_feats_(feats, args['nr_bins'])
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
