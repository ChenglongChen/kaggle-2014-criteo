#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-g', '--high', type=int, default=int(1e+4))
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

def read_freqent_feats(threshold=10):
    frequent_feats = {}
    for row in csv.DictReader(open('fc.trva.r1.p1.t10.txt')):
        if int(row['Total']) < threshold:
            continue
        frequent_feats[row['Field']+'-'+row['Value']] = int(row['Total'])
    return frequent_feats

frequent_feats = read_freqent_feats(args['threshold'])

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats, special_feat = [], ""
        for feat in gen_feats(row):
            field = feat.split('-')[0]
            type, field = field[0], int(field[1:])
            if type == 'C' and feat not in frequent_feats:
                feat = feat.split('-')[0]+'less'
            if type == 'C':
                field += 13
            feats.append((field, feat))
            if feat in frequent_feats and frequent_feats[feat] > args['high']:
                special_feat += feat
        feats.append((40, special_feat))
        feats = gen_hashed_fm_feats(feats, args['nr_bins'])
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
