#!/usr/bin/env python3

import argparse, csv, hashlib, sys

from common import *

def open_with_header_witten(path):
    f = open(path, 'w')
    f.write('Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26\n')
    return f

def parse_args():
    
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    parser = argparse.ArgumentParser(description='split data into days')
    parser.add_argument('csv_path', type=str, help='set path to the csv file')
    args = vars(parser.parse_args())

    return args

args = parse_args()

sizes = [x*100000 for x in [64.5, 125.5, 193.5, 263, 331, 398, -1]]

day = 0
f = open_with_header_witten('{0}.{1}'.format(args['csv_path'], day))
for i, line in enumerate(open_skip_first_line(args['csv_path']), start=1):
    if i == sizes[day]:
        f.close()
        day += 1
        f = open_with_header_witten('{0}.{1}'.format(args['csv_path'], day))
    f.write(line)
f.close()
