#!/usr/bin/env python

import pandas, argparse, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='analyze')
parser.add_argument('csv_path', type=str, help='set path to the csv file')
args = vars(parser.parse_args())

df = pandas.io.parsers.read_csv(args['csv_path'])
df = df.fillna(0)
for i in range(1, 14):
    print('{0:3} {1:>10}'.format('I'+str(i), len(set(df['I{0}'.format(i)]))))
for i in range(1, 27):
    print('{0:3} {1:>10}'.format('C'+str(i), len(set(df['C{0}'.format(i)]))))
