#!/usr/bin/env python3

import csv, itertools, collections

dict = collections.defaultdict(lambda: set())
for row in csv.DictReader(open('fc.trva.t2.txt')):
    dict[row['Field']].add(row['Value'])

for f1, f2 in itertools.combinations(dict.keys(), 2):
    output = '{0:3} {1:10} {2:3} {3:10} {4:10}'.\
        format(f1, len(dict[f1]), f2, len(dict[f2]), len(dict[f1].intersection(dict[f2])))
    print(output)
