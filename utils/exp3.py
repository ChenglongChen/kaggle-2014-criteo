#!/usr/bin/env python3

import subprocess, sys, os, time

cmd = 'utils/prepare.sh'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["100"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer.py -n 48 converters/num.py tr.r{size}.csv tr.r{size}.gbdt'.format(size=size)
    worker_tr = subprocess.Popen(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 24 converters/num.py va.r{size}.csv va.r{size}.gbdt'.format(size=size)
    worker_va = subprocess.Popen(cmd, shell=True) 

    worker_tr.communicate()
    worker_va.communicate()

    cmd = './mark29 -t 30 -s 24 -v va.r{size}.gbdt tr.r{size}.gbdt'.format(size=size) 
    subprocess.call(cmd, shell=True)

    cmd = 'converters/parallelizer.py -n 48 converters/gbdt.py tr.r{size}.csv tr.r{size}.gbdt.out tr.r{size}.fm'.format(size=size)
    worker_tr = subprocess.Popen(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 24 converters/gbdt.py va.r{size}.csv va.r{size}.gbdt.out va.r{size}.fm'.format(size=size)
    worker_va = subprocess.Popen(cmd, shell=True) 

    worker_tr.communicate()
    worker_va.communicate()

    cmd = './mark33 -q -s 192 -t 40 -v va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    subprocess.call(cmd, shell=True) 

print('time used = {0:.0f}'.format(time.time()-start))
