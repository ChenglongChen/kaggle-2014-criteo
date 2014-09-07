#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-z', '--type', type=int, default=0)
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for feat in gen_feats(row):
            field = feat.split('-')[0]
            type, field = field[0], int(field[1:])
            if type == 'C':
                continue
            if args['type'] == 0:
                feats.append(feat)
            else:
                feats.append((field, feat))
        if args['type'] == 0:
            feats = gen_hashed_svm_feats(feats, args['nr_bins'])
        else:
            feats = gen_hashed_fm_feats(feats, args['nr_bins'])
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
