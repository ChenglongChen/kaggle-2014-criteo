#!/usr/bin/env python3

import argparse, csv, hashlib, sys, collections

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='find max and min')
parser.add_argument('path', type=str, help='set path to the svm file')
args = vars(parser.parse_args())

max_values = collections.defaultdict(lambda: float('-inf'))
min_values = collections.defaultdict(lambda: float('inf'))

for i, row in enumerate(csv.reader(open(args['path']))):
    if i == 0:
        continue
    for i, element in enumerate(row):
        if 2 <= i and i <= 14:
            if element == '':
                continue
            value = float(element)
            key = i-1
            min_values[key] = min(value, min_values[key])
            max_values[key] = max(value, max_values[key])

print(max_values)
print(min_values)
