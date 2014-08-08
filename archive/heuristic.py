#!/usr/bin/env python3

import argparse, csv, hashlib, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='scan')
parser.add_argument('csv_path', type=str, help='set path to the csv file')
parser.add_argument('out_path', type=str, help='set path to the csv file')
args = vars(parser.parse_args())

f = csv.writer(open(args['out_path'], 'w'))
for i, row in enumerate(csv.reader(open(args['csv_path']))):
    if i == 0:
        f.writerow(row)
        continue
    prob = float(row[1])
    if 0.02 < prob and prob < 0.6:
        prob -= 0.01
    row[1] = str(prob)
    f.writerow(row)
