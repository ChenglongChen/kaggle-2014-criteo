#!/usr/bin/env python

import sys

if(len(sys.argv) != 5):
	print('Usage: python filtering.py min_count input_libsvm_file counts_file output_libsvm_file')
	exit(0)

threshold = int(sys.argv[1])

counts = set()
with open(sys.argv[3], 'r') as f_in:
	for line in f_in:
		index, count = line.rstrip().split(':')
		if(int(count) >= threshold):
			counts.add(index)
print('nr_feats = {0}'.format(len(counts)))

with open(sys.argv[2], 'r') as f_in, open(sys.argv[4], 'w') as f_out:
	for line in f_in:
		fields = line.rstrip().split()	
		new_fields = [fields[0]]
		for field in fields[1:]:
			index, value = field.split(':')
			if index in counts:
				new_fields.append(field)
		f_out.write(' '.join(new_fields)+'\n')
