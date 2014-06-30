#!/usr/bin/env python3

import argparse, csv, hashlib, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='scan')
parser.add_argument('csv_path', type=str, help='set path to the csv file')
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
    if i % 10000 == 0:
        print(round(float(pos)/(pos+neg), 5))
