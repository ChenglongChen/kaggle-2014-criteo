import sys

if(len(sys.argv) != 5):
	print('Usage: python filtering.py min_count input_libsvm_file counts_file output_libsvm_file')
	exit(0)

counts = dict()
with open(sys.argv[3], 'r') as f_in:
	for line in f_in:
		index, count = line.rstrip().split(':')
		if(int(count) >= int(sys.argv[1])):
			counts[index] = count

with open(sys.argv[2], 'r') as f_in, open(sys.argv[4], 'w') as f_out:
	for line in f_in:
		fields = line.rstrip().split()	
		new_fields = [fields.pop(0)]
		for field in fields:
			index, value = field.split(':')
			if(index in counts.keys()):
				new_fields.append(field)
		f_out.write(' '.join(new_fields)+'\n')
