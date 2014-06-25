#!/usr/bin/env python3

import argparse, csv, hashlib

parser = argparse.ArgumentParser(description='process some integers')
parser.add_argument('-n', type=int, nargs=1, default=100, help='set number of bins for hashing trick')
parser.add_argument('csv_path', type=str, nargs=1, help='set path to the csv file')
parser.add_argument('svm_path', type=str, nargs=1, help='set path to the svm file')
args = parser.parse_args()

NR_BINS, CSV_PATH, SVM_PATH = args.n, args.csv_path[0], args.svm_path[0]

def hashstr(str):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%NR_BINS

with open(SVM_PATH, 'w') as f:
    for i, row in enumerate(csv.reader(open(CSV_PATH))):
        if i == 0:
            continue
        output = ''
        for i, element in enumerate(row):
            if i == 0:
                continue
            elif i == 1:
                output += element 
            elif i <= 14:
                if element == '':
                   element = '0' 
                output += ' {0}:{1}'.format(i-1, element)
            else:
                bin = hashstr(str(i)+element)+40 if element != '' else i-1
                output += ' {0}:1'.format(bin)
        output += '\n'
        f.write(output)
