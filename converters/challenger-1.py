#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-f', '--fields', type=str, default="")
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

if len(args['fields'].split(',')) != 2:
    print('error')
    exit(1)

FIELDS = map(int, args['fields'].split(','))
I_FIELDS, C_FIELDS = [], []
for field in FIELDS:
    if field <= 13:
        I_FIELDS.append(field)
    else:
        C_FIELDS.append(field)

def gen_feats(row):
    feats = []

    for j in I_FIELDS:
        field = 'I{0}'.format(j)
        value = row[field]
        if j == 5 and value != '':
            value = int(math.log(float(value)+1))
        elif j in [2, 3, 6, 7, 9] and value != '':
            value = int(float(value)/10)
        key = field + '-' + str(value)
        feats.append(key)

    for j in C_FIELDS:
        field = 'C{0}'.format(j)
        value = row[field]
        key = field + '-' + value
        feats.append(key)

    return feats

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for feat in gen_feats(row):
            field = feat.split('-')[0]
            type, field = field[0], int(field[1:])
            if type == 'C':
                field += 13
            feats.append((field, feat))
        feats = gen_hashed_fm_feats(feats, args['nr_bins'])
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
