#!/usr/bin/env python3

import sys, argparse, re, itertools

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, default=0.55)
parser.add_argument('log_path', type=str)
args = vars(parser.parse_args())

useful_pairs = set()

for line in open(args['log_path']):
    if line.startswith('f1'):
        f1, f2 = map(int, re.findall(r'f1 = (.*), f2 = (.*)', line)[0])
    elif line.startswith('  9'):
        logloss = float(line.strip().split()[-1])
        if logloss < args['threshold']:
            useful_pairs.add((f1, f2))

sys.stdout.write("#include <vector>\n")
sys.stdout.write("std::vector<int> useful_pairs = {")

useful_pairs_ = []
for f1, f2 in itertools.combinations(range(1, 40), 2):
    useful = 1 if (f1, f2) in useful_pairs else 0
    useful_pairs_.append(str(useful))
sys.stdout.write(','.join(useful_pairs_))

sys.stdout.write('};\n')