#!/usr/bin/env python3

import csv, subprocess

workers = []
for size in [100]:
    for threshold in [1, 10, 100, 1000]:
        for data in ['tr', 'va']:
            src = '{data}.r{size}.csv'.format(data=data, size=size)
            dst = '{data}.r{size}.t{threshold}.csv'.\
                format(data=data, size=size, threshold=threshold)
            if threshold == 1:
                cmd = 'ln -sf {src} {dst}'.format(src=src, dst=dst)
            else:
                cmd = './filters/filter.py -t {threshold} {src} {dst}'.\
                    format(threshold=threshold, src=src, dst=dst)
            worker = subprocess.Popen(cmd, shell=True)
            workers.append(worker)

for worker in workers:
    worker.communicate()
