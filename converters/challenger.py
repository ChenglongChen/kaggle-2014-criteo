#!/usr/bin/env python3

import argparse, csv, sys, math

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('csv_path', type=str)
parser.add_argument('svm1_path', type=str)
parser.add_argument('svm2_path', type=str)
args = vars(parser.parse_args())

GROUP1 = ['3', '12', '16']

frequent_feats = read_freqent_feats(10)

def write(label, feats, f, coef):
    feats = gen_hashed_svm_feats(feats, args['nr_bins'], coef)
    f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')

with open(args['svm1_path'], 'w') as f1, open(args['svm2_path'], 'w') as f2:
    for row in csv.DictReader(open(args['csv_path'])):
        raw_feats = gen_feats(row)

        feats1, feats2 = [], []
        for feat in raw_feats:
            if feat.startswith('I'):
                feats2.append(feat)
            elif feat.startswith('C'):
                field = feat.split('-')[0][1:]
                if feat not in frequent_feats:
                    feat = 'C'+field+'less'
                if field in GROUP1:
                    feats1.append(feat)
                else:
                    feats2.append(feat)
            else:
                raise ValueError

        coef = 1/math.sqrt(float(len(raw_feats)))
        write(row['Label'], feats1, f1, coef)
        write(row['Label'], feats2, f2, coef)
