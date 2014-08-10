#!/usr/bin/env python3

import argparse, csv, sys, itertools, collections

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=int, default=int(100))
parser.add_argument('-p', '--poly2_threshold', type=int, default=int(100))
parser.add_argument('csv_path', type=str)
args = vars(parser.parse_args())

valid_feats = set()
for row in csv.DictReader(open('logs/trva.feature_counts.t{0}.log'.format(args['threshold']))):
    valid_feats.add(row['Field']+'-'+row['Value'])

poly2_counts = collections.defaultdict(lambda : [0, 0, 0])
for row in csv.DictReader(open(args['csv_path'])):
    feats = gen_feats(row)
    for feat1, feat2 in itertools.combinations(feats, 2):
        if feat1 not in valid_feats or feat2 not in valid_feats:
            continue
        feat = feat1 + '-' + feat2
        if row['Label'] == '0':
            poly2_counts[feat][0] += 1
        else:
            poly2_counts[feat][1] += 1
        poly2_counts[feat][2] += 1

print('Key,Neg,Pos,Total,Ratio')
for key, (neg, pos, total) in sorted(poly2_counts.items(), key=lambda x: x[1][2]):
    if total < args['poly2_threshold']:
        continue
    ratio = round(float(pos)/total, 5)
    print(key+','+str(neg)+','+str(pos)+','+str(total)+','+str(ratio))
