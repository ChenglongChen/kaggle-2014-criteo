#!/usr/bin/env python

import argparse, csv, hashlib, sys, collections

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--accuracy', type=float, default=0.01)
parser.add_argument('svm_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

records = collections.defaultdict(lambda : [0, 0, 0.0])
for line_svm, line_out in zip(open(args['svm_path']), open(args['out_path'])):
    label = line_svm.rstrip().split(' ', 1)[0]
    prob = float(line_out.rstrip())
    bin = int(prob/args['accuracy'])
    if label == '1':
        records[bin][0] += 1
    else:
        records[bin][1] += 1
    records[bin][2] += prob

x, diffs = [], []
for bin, (pos, neg, prob) in records.items():
    total = pos+neg
    diff = (prob-float(pos))/total
    print('{0:4.2f} {1:7d} {2:7d} {3:7d} {4:9.3f}'.format(bin*args['accuracy'], pos, int(prob), total, diff)) 
    x.append(bin*args['accuracy'])
    diffs.append(diff)
