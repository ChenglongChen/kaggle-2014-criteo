#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

target_cat_feats = ['C20-', 'C19-']

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for j in range(1, 14):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10 
            feats.append('{0}:{1}'.format(j, val))
        
        cat_feats = set()
        for j in range(1, 27):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            cat_feats.add(key)

        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append('{0}:{1}'.format(13+j, 1))
            else:
                feats.append('{0}:{1}'.format(13+j, 0))

        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
