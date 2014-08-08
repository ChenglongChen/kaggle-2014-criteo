#!/usr/bin/env python3

import argparse, csv, hashlib, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='scan')
parser.add_argument('csv_path', type=str, help='set path to the csv file')
parser.add_argument('-n', type=int, help='set path to the csv file')
args = vars(parser.parse_args())

pos, neg = 0, 0

for i, row in enumerate(csv.reader(open(args['csv_path']))):
    if i == 0:
        continue
    i += 1
    if row[1] == '1':
        pos += 1
    else:
        neg += 1
    if i % args['n'] == 0:
        print(round(float(pos)/(pos+neg), 5))
        pos = 0
        neg = 0
