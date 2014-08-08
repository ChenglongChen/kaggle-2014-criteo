#!/usr/bin/env python3

import argparse, csv, hashlib, sys, collections

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('-a', '--special_field', type=int, default=0)
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

last_values = collections.defaultdict(lambda : [0 for i in range(13)])
with open(args['svm_path'], 'w') as f:
    for row in open_csv_skip_first_line(args['csv_path']):
        feats = set()
        for idx, element in enumerate(row[2:15], start=1):
            bin = hashstr(str(idx)+str(element), args['nr_bins'])
            feats.add((bin, 1))

        for idx, element in enumerate(row[15:], start=1):
            bin = hashstr(str(idx+20)+element, args['nr_bins'])
            feats.add((bin, 1))
            if idx != args['special_field']:
                continue
            for idx2, element2 in enumerate(row[2:15]):
                if element2 == '':
                    continue
                element2 = int(element2)
                prev_value =  last_values[element][idx2]
                last_values[element][idx2] = element2
                element2 = element2 - prev_value
                feats.add((args['nr_bins']+idx2+100, element2/100.0))

        feats = list(feats)
        feats.sort()
        feats = ['{0}:{1}'.format(idx, val) for (idx, val) in feats]
        f.write(row[1] + ' ' + ' '.join(feats) + '\n')
