#!/usr/bin/env python

import pandas, argparse, sys, math

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='analyze')
parser.add_argument('csv_path', type=str, help='set path to the csv file')
args = vars(parser.parse_args())

df = pandas.io.parsers.read_csv(args['csv_path'])
#for i in range(1, 14):
#    v = df['I{0}'.format(i)]
#    target = float('nan')
#    idx = 0
#    while math.isnan(target):
#        target = v[idx] 
#        idx += 1
#    print(i, target)
#    df1 = df[v == target]
#    print(df1)

for i in range(1, 27):
    v = df['C{0}'.format(i)]
    target = 'nan'
    idx = 0
    while str(target) == 'nan':
        target = v[idx] 
        idx += 1
    print(i, target)
    df1 = df[v == target]
    print(df1)
