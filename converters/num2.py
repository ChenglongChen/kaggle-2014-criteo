#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-c', '--cat_feat', type=int, default=int(1))
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

frequent_feats = read_freqent_feats(args['threshold'])

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for j in range(1, 14):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10 
            feats.append('{0}:{1}'.format(j, val))
        
        field = 'C{0}'.format(args['cat_feat'])
        key = field + '-' + row[field]
        feats.append('{0}:{1}'.format(14, frequent_feats[key]))
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
