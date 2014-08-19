#!/usr/bin/env python3

import sys, argparse, re

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, default=0.55)
parser.add_argument('log_path', type=str)
args = vars(parser.parse_args())

useful_pairs = []

for line in open(args['log_path']):
    if line.startswith('f1'):
        f1, f2 = re.findall(r'f1 = (.*), f2 = (.*)', line)[0]
    elif line.startswith('  9'):
        logloss = float(line.strip().split()[-1])
        if logloss < args['threshold']:
            useful_pairs.append((f1, f2))

for f1, f2 in useful_pairs:
    print('{0} {1}'.format(f1, f2))
