#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

target_cat_feats = ['C9-a73ee510', 'C22-', 'C5-25c83c98', 'C8-0b153874', 'C1-05db9164', 'C17-e5ba7672', 'C26-', 'C25-', 'C19-', 'C20-', 'C23-32c7478e', 'C6-7e0ccccf', 'C14-b28479f6', 'C19-21ddcdc9', 'C14-07d13a8f', 'C10-3b08e48b', 'C6-fbad5c96', 'C23-3a171ecb', 'C20-b1252a9d', 'C20-5840adea', 'C6-fe6b92e5', 'C20-a458ea53']

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
