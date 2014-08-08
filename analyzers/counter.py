#!/usr/bin/env python3

import argparse, csv, sys, collections

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
args = vars(parser.parse_args())

counts = collections.defaultdict(lambda : [0, 0, 0])

for row in csv.DictReader(open(args['csv_path'])):
    label = row['Label']
    for j in range(1, 27):
        field = 'C{0}'.format(j)
        value = row[field]
        if label == '0':
            counts[field+','+value][0] += 1
        else:
            counts[field+','+value][1] += 1
        counts[field+','+value][2] += 1

print('Field,Value,Neg,Pos,Total')
for key, (neg, pos, total) in sorted(counts.items(), key=lambda x: x[1][2]):
    print(key+','+str(neg)+','+str(pos)+','+str(total))
