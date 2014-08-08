#!/usr/bin/env python3

import argparse, sys, math

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='process some integers')
parser.add_argument('va_path', type=str, help='set path to the svm file')
parser.add_argument('out_path', type=str, help='set path to the svm file')
args = vars(parser.parse_args())

log_loss, counter = 0, 0
for line1, line2 in zip(open(args['va_path']), open(args['out_path'])):
    line1 = line1.strip().split()
    label = line1[0]
    prob = float(line2.strip())

    if label == '1':
        if prob <= 0:
            prob = 0.001
        if prob > 1:
            prob = 1
        log_loss += math.log(prob)
    else:
        if prob >= 1:
            prob = 0.999
        if prob < 0:
            prob = 0
        log_loss += math.log(1-prob)

    counter += 1

print(-log_loss/counter)
