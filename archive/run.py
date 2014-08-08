#!/usr/bin/env python3

import argparse, csv, hashlib, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('-a', '--accuracy', type=float, default=0.01)
parser.add_argument('svm_path', type=str)
parser.add_argument('out_path', type=str)
parser.add_argument('new_svm_path', type=str)
args = vars(parser.parse_args())

with open(args['new_svm_path'], 'w') as f:
    for i, (line_svm, line_out) in enumerate(zip(open(args['svm_path']), open(args['out_path']))):
        prob = float(line_out.rstrip())
        bin = int(prob/args['accuracy'])

        f.write(line_svm.rstrip() + ' {0}:1'.format(args['nr_bins']+bin) + '\n')

#import argparse, sys, collections
#
#if len(sys.argv) == 1:
#    sys.argv.append('-h')
#
#parser = argparse.ArgumentParser(description='process some integers')
#parser.add_argument('tr_path', type=str, help='set path to the svm file')
##parser.add_argument('va_path', type=str, help='set path to the svm file')
#args = vars(parser.parse_args())
#
#records = collections.defaultdict(lambda: [0,0])
#for line in open(args['tr_path']):
#    line = line.strip().split()
#    label = line[0]
#    for feat in line[1:]:
#        idx, val = feat.split(':')
#        if label == '1':
#            records[idx][0] += 1 
#        else:
#            records[idx][1] += 1 
#
#freqencies = {}
#for key, val in records.items():
#    total = val[0] + val[1]
#    if total < 10000:    
#        continue
#    freqencies[key] = ((float(val[0]) / total), total)
#
#for key, val in sorted(freqencies.items(), key=lambda x: x[1][0]):
#    print(key, val[0], val[1])

#def write_file(path):
#    with open(path+'.tmp', 'w') as f:
#        for line in open(path):
#            line = line.strip().split()
#            f.write(line[0])
#            for feat in line[1:]:
#                idx, val = feat.split(':')
#                f.write(' {0}:1'.format(idx))
#            for feat in line[1:]:
#                idx, val = feat.split(':')
#                if idx not in freqencies:
#                    continue
#                f.write(' {0}:{1}'.format(int(idx)+10000000, round(freqencies[idx],3)))
#            f.write(' 20000500:1\n')
#
#write_file(args['tr_path'])
#write_file(args['va_path'])