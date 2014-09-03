#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
	sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+8))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('csv_path', type=str)
parser.add_argument('svm_path', type=str)
args = vars(parser.parse_args())

frequent_feats = read_freqent_feats(args['threshold'])

def gen_feats_(row):
	feats = []
	for j in range(1, 14):
		field = 'I{0}'.format(j)
		value = row[field]
		v1 = None
		v2 = None
		if value != '':
			value = int(value)
			if value >= 2:
				v1 = int(math.log(float(value))**2)
				#v2 = int(math.log(float(value)))
			else:
				v1 = 'SP'+str(value)
				#v2 = 'SP'+str(value)
		
		#key = field + '-' + str(v1) + 'sep' + str(v2)
		key = field + '-v1' + str(v1)
		new_field = 'I{0}'.format(j+39)
		#key = field + '-' + str(v2)
		feats.append(key)
	
	for j in [5]:
		field = 'I{0}'.format(j)
		value = row[field]
		if j == 5 and value != '':
			q = int(int(value)/1437)
			r = int(value)%1437
			feats.append('I53' + '-p5r1437log' + str(int(math.log(r+1))))
			feats.append('I54' + '-p5r1437log^2' + str(int(math.log(r+1)**2)))
			#feats.append(field + '-p5r1437log-1.3^2' + str(int((math.log(r+1)-1.3)**2)))
			feats.append('I55' + '-p5q1437log' + str(int(math.log(q+1))))
			feats.append('I56' + '-p5q1437log^2' + str(int(math.log(q+1)**2)))

	for j in range(1, 14):
		if j in [1]:
			continue
		field = 'I{0}'.format(j)
		value = row[field]
		if j == 5 and value != '':
			value = int(math.log(float(value)+1))
		elif j in [2, 3, 6, 7, 9] and value != '':
			value = int(float(value)/10)
		key = field + '-prev' + str(value)
		feats.append(key)

	for j in range(1, 27):
		field = 'C{0}'.format(j)
		value = row[field]
		key = field + '-' + value
		feats.append(key)
	return feats

with open(args['svm_path'], 'w') as f:
	for row in csv.DictReader(open(args['csv_path'])):
		feats = []
		for feat in gen_feats_(row):
			field = feat.split('-')[0]
			type, field = field[0], int(field[1:])
			if type == 'C' and feat not in frequent_feats:
				feat = feat.split('-')[0]+'less'
			if type == 'C':
				field += 13
			#feats.append((field, feat)) # for fac. model
			feats.append(feat) # for svm
		#feats = gen_hashed_fm_feats(feats, args['nr_bins']) # for fac. model
		feats = gen_hashed_svm_feats(feats, args['nr_bins']) # for fac. model
		f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
