#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

from common import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('fm_path', type=str)
parser.add_argument('gbdt_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

with open(args['out_path'], 'w') as f:
    for line_fm, line_gbdt in zip(open(args['fm_path']), open(args['gbdt_path'])):

        tokens_fm = line_fm.strip().split()
        tokens_gbdt = line_gbdt.strip().split()

        label, tokens_fm = tokens_fm[0], tokens_fm[1:]
        tokens_gbdt = tokens_gbdt[1:]

        val = round(1/math.sqrt(float(len(tokens_fm)+len(tokens_gbdt))), 5)

        line_fm_2 = [] 

        for token in tokens_fm:
            field, idx, dummy = token.split(':')
            line_fm_2.append('{0}:{1}:{2}'.format(field, idx, val))

        for token in tokens_gbdt:
            field, node_idx = map(int, token.split(':'))
            field += 39
            idx = hashstr(token, args['nr_bins'])
            line_fm_2.append('{0}:{1}:{2}'.format(field, idx, val))

        f.write(label+' '+' '.join(line_fm_2)+'\n')
