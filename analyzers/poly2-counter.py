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
for i, row in enumerate(csv.DictReader(open(args['csv_path'])), start=1):
    feats = gen_feats(row)
    tmp = []
    for feat in feats:
        if feat not in valid_feats:
            continue
        tmp.append(feat)
    feats = tmp
    for feat1, feat2 in itertools.combinations(feats, 2):
        feat = feat1 + '-' + feat2
        if row['Label'] == '0':
            poly2_counts[feat][0] += 1
        else:
            poly2_counts[feat][1] += 1
        poly2_counts[feat][2] += 1
    if i % 100000 == 0:
        sys.stderr.write('{0}k\n'.format(int(i/1000)))

tmp_dict = {}
for key, (neg, pos, total) in poly2_counts.items():
    if total < args['poly2_threshold']:
        continue
    tmp_dict[key] = [neg, pos, total]
poly2_counts = tmp_dict

print('Key,Neg,Pos,Total,Ratio')
for key, (neg, pos, total) in sorted(poly2_counts.items(), key=lambda x: x[1][2]):
    ratio = round(float(pos)/total, 5)
    print(key+','+str(neg)+','+str(pos)+','+str(total)+','+str(ratio))
