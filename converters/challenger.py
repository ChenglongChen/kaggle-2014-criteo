#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+4))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

FieldSizeI = list(map(lambda x: int(10**x)*args['nr_bins'], [0 for i in range(13)]))
FieldSizeC = list(map(lambda x: int(10**x)*args['nr_bins'],  
    [1, 0, 2, 2, 0, 0, 2, 0, 0, 2,
     1, 2, 1, 0, 1, 2, 0, 1, 1, 0, 
     2, 0, 0, 2, 0, 2]))
FieldSize = FieldSizeI+FieldSizeC
print('nr_bins = {0}'.format(sum(FieldSize)))

frequent_feats = read_freqent_feats(args['threshold'])

def gen_hashed_fm_feats_(feats, coef=None):
    fm_feats = []
    for field, feat in feats:
        nr_bins = FieldSize[field-1]
        feat = hashstr(feat, nr_bins)
        fm_feats.append((field, feat))
    fm_feats.sort()
    if coef is not None:
        val = coef
    else:
        val = 1/math.sqrt(float(len(fm_feats)))
    fm_feats = ['{0}:{1}:{2}'.format(field, idx, val) for (field, idx) in fm_feats]
    return fm_feats

def gen_feats_(row):
    feats = []
    for j in range(1, 14):
        field = 'I{0}'.format(j)
        value = row[field]
        if j == 5 and value != '':
            value = int(math.log(float(value)+1)**2)
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

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for feat in gen_feats_(row):
            field = feat.split('-')[0]
            type, field = field[0], int(field[1:])
            if type == 'C' and feat not in frequent_feats:
                feat = feat.split('-')[0]+'less'
            if type == 'C':
                field += 13
            feats.append((field, feat))
        feats = gen_hashed_fm_feats_(feats)
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
