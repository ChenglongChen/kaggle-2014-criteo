#!/usr/bin/env python3

import subprocess, sys, os, time

cmd = 'git clean -df && utils/prepare.sh'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["100", "10", "1"]:
    print('size = {size}'.format(size=size))

    for data in ['tr', 'va']:
        cmd = 'converters/parallelizer.py -n 24 converters/num.py {data}.r{size}.csv {data}.r{size}.svm.num'.format(size=size, data=data)
        subprocess.call(cmd, shell=True)

    for data in ['tr', 'va']:
        cmd = 'converters/parallelizer.py -n 24 converters/defender.py {data}.r{size}.csv {data}.r{size}.fm'.format(size=size, data=data)
        subprocess.call(cmd, shell=True)

    cmd = './mark29 -s 24 -v va.r{size}.svm.num tr.r{size}.svm.num'.format(size=size) 
    subprocess.call(cmd, shell=True)

    for data in ['tr', 'va']:
        cmd = 'converters/parallelizer.py -n 24 converters/combine.py {data}.r{size}.fm {data}.r{size}.svm.num.gbdt {data}.r{size}.fm2'.format(size=size, data=data)
        subprocess.call(cmd, shell=True)

    cmd = './fm-train -u 3 -q -s 24 -t 20 -v va.r{size}.fm2 tr.r{size}.fm2'.format(size=size) 
    subprocess.call(cmd, shell=True) 

print('time used = {0}'.format(time.time()-start))
