#!/usr/bin/env python

import sys
from collections import defaultdict

if(len(sys.argv) != 3):
	print('Usage: python counting.py input_libsvm_file output_counts_file')
	exit(0)

counts = defaultdict(lambda: 0)

with open(sys.argv[1], 'r') as f_in:
	for line in f_in:
		fields = line.rstrip().split()
		del fields[0]
		for field in fields:
			index, value = field.split(':')
			counts[int(index)] += 1

with open(sys.argv[2], 'w') as f_out:
	for key in sorted(counts.keys()):
		f_out.write('{0}:{1}\n'.format(key, counts[key]))
