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

def gen_full_feats(row):
    feats = []
    for j in range(1, 14):
        field = 'I{0}'.format(j)
        value = row[field]
        if j == 5 and value != '':
            value = int(math.log(float(value)+1))
        elif j in [2, 3, 6, 7, 9] and value != '':
            value = int(float(value)/10)
        key = field + '-' + str(value)
        feats.append(key)
    for j in range(1, 27):
        field = 'C{0}'.format(j)
        value = row[field]
        key = field + '-' + value
        feats.append(key)
    return feats

valid_feats = set()
for row in csv.DictReader(open('logs/trva.feature_counts.t{0}.log'.format(args['threshold']))):
    valid_feats.add(row['Field']+'-'+row['Value'])

poly2_counts = collections.defaultdict(lambda : [0, 0, 0])
for i, row in enumerate(csv.DictReader(open(args['csv_path'])), start=1):
    feats = gen_full_feats(row)
    tmp = []
    for feat in feats:
        if feat.startswith('C') and feat not in valid_feats:
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
        sys.stderr.write('{0}k, {1}k\n'.format(int(i/1000), int(len(poly2_counts)/1000)))

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
