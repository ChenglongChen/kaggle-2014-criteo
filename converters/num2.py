#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
args = vars(parser.parse_args())

nr_bins = int(10**4)
def gen_hashed_gbdt_feats(feats):
    feats = [(field, hashstr(feat, nr_bins)) for (field, feat) in feats]
    feats.sort()
    feats = ['{0}'.format(idx) for (field, idx) in feats]
    return feats

frequent_feats = read_freqent_feats(10000)

with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for j in range(1, 14):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10 
            feats.append('{0}'.format(val))
        f_d.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
        
        feats = []
        for j in range(1, 27):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            if key not in frequent_feats:
                feats.append((j, field+'less'))
            else:
                feats.append((j, key))
        feats = gen_hashed_gbdt_feats(feats)

        f_s.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
