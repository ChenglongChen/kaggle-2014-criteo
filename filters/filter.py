#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=int, default=int(100))
parser.add_argument('src_path', type=str)
parser.add_argument('dst_path', type=str)
args = vars(parser.parse_args())

stats = {}
for row in csv.DictReader(open('logs/trva.feature_counts.t{0}.log'.format(args['threshold']))):
    stats[row['Field']+row['Value']] \
        = {'Total': int(row['Total']), 'Ratio': float(row['Ratio'])}

f = csv.DictWriter(open(args['dst_path'], 'w'), HEADER.split(','))
f.writeheader()
for row in csv.DictReader(open(args['src_path'])):
    for j in range(1, 27):
        field = 'C{0}'.format(j)
        key = field + row[field]
        if key not in stats:
            row[field] = ''
    f.writerow(row)
