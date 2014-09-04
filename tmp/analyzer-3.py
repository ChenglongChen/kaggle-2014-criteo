#!/usr/bin/env python3

import argparse, csv, sys, collections

from common import *

records = collections.defaultdict(lambda: [0, 0, 0])
for row in csv.DictReader(open('tr.r100.csv')):
    sp_feat = []
    sp_feat.append('C9' +'-'+row['C9'])
    sp_feat.append('C17'+'-'+row['C17'])
    sp_feat.append('C23'+'-'+row['C23'])
    sp_feat = '--'.join(sp_feat)
    if row['Label'] == '1':
        records[sp_feat][1] += 1
    else:
        records[sp_feat][0] += 1
    records[sp_feat][2] += 1

for key, (pos, neg, total) in sorted(records.items(), key=lambda x: x[1][2]):
    print('{0:20} {1:10} {2:10} {3:10}'.format(key, pos, neg, total))
