#!/usr/bin/env python3

import argparse, sys, collections

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='process some integers')
parser.add_argument('tr_path', type=str, help='set path to the svm file')
parser.add_argument('va_path', type=str, help='set path to the svm file')
args = vars(parser.parse_args())

records = collections.defaultdict(lambda: [0,0])
for line in open(args['tr_path']):
    line = line.strip().split()
    label = line[0]
    for feat in line[1:]:
        idx, val = feat.split(':')
        if label == '1':
            records[idx][0] += 1 
        else:
            records[idx][1] += 1 

useful_feat = set()
for key, val in records.items():
    useful_feat.add(key) 

def write_file(path):
    with open(path+'.tmp', 'w') as f:
        for line in open(args['tr_path']):
            line = line.strip().split()
            f.write(line[0])
            for feat in line[1:]:
                idx, val = feat.split(':')
                if idx not in useful_feat:
                    continue
                f.write(' '+idx+':1')
            f.write(' 10000500:1\n')

write_file(args['tr_path'])
write_file(args['va_path'])
