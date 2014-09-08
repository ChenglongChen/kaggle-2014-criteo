#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+7))
parser.add_argument('fm_path', type=str)
parser.add_argument('gbdt_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

with open(args['out_path'], 'w') as f:
    for line_fm, line_gbdt in zip(open(args['fm_path']), open(args['gbdt_path'])):

        tokens = line_gbdt.strip().split()

        val = round(1/math.sqrt(float(len(tokens))), 5)

        line_fm_2 = [] 
        for token in line_gbdt.split():
            idx, node = map(int, token.split(':'))
            line_fm_2.append('{0}:{1}:{2}'.format(idx+39, args['nr_bins']+idx*1000+node, val))

        f.write(line_fm.strip()+' '.join(line_fm_2)+'\n')
