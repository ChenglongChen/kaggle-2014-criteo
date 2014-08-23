#!/usr/bin/env python

import argparse, sys, collections, math

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
        records[bin][2] -= math.log(prob)
    else:
        records[bin][1] += 1
        records[bin][2] -= math.log(1-prob)

x, diffs = [], []
for bin, (pos, neg, loss) in records.items():
    prob = bin*args['accuracy']
    if prob == 0 or prob == 1:
        continue
    prob += 0.5*args['accuracy']
    total = pos+neg
    ideal = -(prob*math.log(prob)+(1-prob)*math.log(1-prob))
    print('{0:4.2f} {1:7.3f} {2:7.3f} {3:6d}'.format(prob, loss/total, ideal, total)) 