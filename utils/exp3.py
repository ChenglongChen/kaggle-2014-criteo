#!/usr/bin/env python3

import subprocess, sys, os, time

cmd = 'git clean -df && utils/prepare.sh && ./converters/dump.py'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["x", "1"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer.py -n 96 converters/num.py tr.r{size}.csv tr.r{size}.gbdt'.format(size=size)
    subprocess.call(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 96 converters/num.py va.r{size}.csv va.r{size}.gbdt'.format(size=size)
    subprocess.call(cmd, shell=True) 

    cmd = './mark29 -t 30 -s 13 va.r{size}.gbdt tr.r{size}.gbdt'.format(size=size) 
    subprocess.call(cmd, shell=True)

    cmd = 'converters/parallelizer.py -n 96 converters/gbdt.py tr.r{size}.csv tr.r{size}.gbdt.out tr.r{size}.fm'.format(size=size)
    subprocess.call(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 96 converters/gbdt.py va.r{size}.csv va.r{size}.gbdt.out va.r{size}.fm'.format(size=size)
    subprocess.call(cmd, shell=True) 

    cmd = './mark33 -s 220 -t 30 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    subprocess.call(cmd, shell=True) 

print('time used = {0:.0f}'.format(time.time()-start))
