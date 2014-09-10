#!/usr/bin/env python3

import argparse, csv, sys, numpy

from common import *

if len(sys.argv) == 1:
	sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
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
		new_field = 'I{0}'.format(j+39)
		key = new_field + '-v1' + str(v1)
		#key = field + '-' + str(v2)
		feats.append(key)
	
	group_a = [6]
	group_b = [1,2,3,4,7,8,9,10,12,13]
	for j1 in group_a:
		for j2 in group_b:
			field1 = 'I{0}'.format(j1)
			field2 = 'I{0}'.format(j2)
			if row[field1] != '' and row[field2] != '':
				if int(row[field1]) >=0 and int(row[field2]) >= 0:
					va = int(row[field1])# I6
					vb = float(row[field2])
					v3 = int(math.log((va+1)/(vb+1))*4)
					new_field = 'I{0}'.format(j2+52)
					feats.append(field1 + '-' + field2 + '-divide-' + str(v3))

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
			feats.append((field, feat)) # for fac. model
			#feats.append(feat) # for svm
		feats = gen_hashed_fm_feats(feats, args['nr_bins']) # for fac. model
		#feats = gen_hashed_svm_feats(feats, args['nr_bins']) # for fac. model
		f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')