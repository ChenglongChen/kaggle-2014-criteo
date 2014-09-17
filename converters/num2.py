#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

valid_features = read_freqent_feats(10000)

def read_pseudo_ctr():
    pseudo_ctr = {}
    for row in csv.DictReader(open('fc.tr.t10.txt')):
        key = row['Field'] + '-' + row['Value']
        if key not in valid_features:
            continue
        pseudo_ctr[key] = row['Ratio']
    return pseudo_ctr

pseudo_ctr = read_pseudo_ctr()

with open(args['svm_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for j in range(1, 14):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10 
            feats.append('{0}:{1}'.format(j, val))
        
        for j in range(1, 27):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            if key in pseudo_ctr:
                feats.append('{0}:{1}'.format(13+j, pseudo_ctr[key]))
            else:
                feats.append('{0}:{1}'.format(13+j, -1))

        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
