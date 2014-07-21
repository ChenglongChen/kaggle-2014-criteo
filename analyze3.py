#!/usr/bin/env python3

import argparse, csv, hashlib, sys, collections

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=int, default=1000)
parser.add_argument('csv_path', type=str)
args = vars(parser.parse_args())

records = collections.defaultdict(lambda : [0, 0, 0])
for i, row in enumerate(open_csv_skip_first_line(args['csv_path'])):
    label = row[1]

    for i, element in enumerate(row[2:15], start=1):
        if label == '0':
            records['I'+str(i)+':'+str(element)][0] += 1
        else:
            records['I'+str(i)+':'+str(element)][1] += 1

    for i, element in enumerate(row[15:], start=1):
        if label == '0':
            records['C'+str(i)+':'+str(element)][0] += 1
        else:
            records['C'+str(i)+':'+str(element)][1] += 1

records_filterd = []
for key in records:
    neg = records[key][0]
    pos = records[key][1]
    total = neg+pos
    if total < args['threshold']:
        continue 
    rate = float(pos)/total
    records_filterd.append((key, rate, total))

records_filterd.sort(key=lambda x: x[1])

for record in records_filterd:
    print('{0:15}{1:10.3f} ({2})'.format(record[0], record[1], record[2]))
