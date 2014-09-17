#!/usr/bin/env python

import argparse, csv, hashlib, sys, collections

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

records = collections.defaultdict(lambda : [0, 0, 0.0])
for line_csv, line_out in zip(csv.DictReader(open(args['csv_path'])), open(args['out_path'])):
    label = line_csv['Label']
    prob = float(line_out.rstrip())
    bin = int(prob*100)
    if label == '1':
        records[bin][0] += 1
    else:
        records[bin][1] += 1
    records[bin][2] += prob

for bin, (pos, neg, prob) in records.items():
    total = pos+neg
    diff = (prob-float(pos))/total
    print('{0:4d} {1:7d} {2:7d} {3:7d} {4:9.3f}'.format(bin, pos, int(prob), total, diff)) 
