#!/usr/bin/env python3

import argparse, csv, hashlib, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

FieldSizesI = [100000,  100000,  100000,  100000,  1000000, 100000,  100000,  100000,  100000,  100000,  100000,  100000,  100000]
FieldSizesC = [100000,  100000,  1000000, 1000000, 100000,  100000,  1000000, 100000,  100000,  1000000, 100000,  1000000, 100000,  100000,  1000000, 1000000, 100000,  100000,  100000,  100000,  1000000, 100000,  100000,  1000000, 100000,  1000000]

with open(args['svm_path'], 'w') as f:
    for i, row in enumerate(open_csv_skip_first_line(args['csv_path'])):
        feats = []
        for i, element in enumerate(row[2:15]):
            bin = hashstr(str(i)+str(element), FieldSizesI[i]-1)+1
            feats.append((bin, 1))

        for i, element in enumerate(row[15:]):
            if element == '':
                bin = hashstr(str(i+14), FieldSizesC[i]-1)+1
            else:
                bin = int(element, 32)%(FieldSizesC[i]-1)+1
            feats.append((bin, 1))

        #feats = list(feats)
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(row[1] + ' ' + ' '.join(feats) + '\n')
