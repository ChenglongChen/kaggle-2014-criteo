#!/usr/bin/env python3

import subprocess, sys, os, time

cmd = 'git clean -df && utils/prepare.sh'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["100", "10", "1"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer.py -n 48 converters/num.py tr.r{size}.csv tr.r{size}.svm.num'.format(size=size)
    worker_tr = subprocess.Popen(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 24 converters/num.py va.r{size}.csv va.r{size}.svm.num'.format(size=size)
    worker_va = subprocess.Popen(cmd, shell=True) 

    worker_tr.communicate()
    worker_va.communicate()

    cmd = 'converters/parallelizer.py -n 48 converters/defender.py tr.r{size}.csv tr.r{size}.fm'.format(size=size)
    worker_tr = subprocess.Popen(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 24 converters/defender.py va.r{size}.csv va.r{size}.fm'.format(size=size)
    worker_va = subprocess.Popen(cmd, shell=True) 

    cmd = './mark29 -t 30 -s 12 va.r{size}.svm.num tr.r{size}.svm.num'.format(size=size) 
    subprocess.call(cmd, shell=True)

    worker_tr.communicate()
    worker_va.communicate()

    cmd = 'converters/parallelizer.py -n 48 converters/combine.py tr.r{size}.fm tr.r{size}.svm.num.out tr.r{size}.fm2'.format(size=size)
    worker_tr = subprocess.Popen(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 24 converters/combine.py va.r{size}.fm va.r{size}.svm.num.out va.r{size}.fm2'.format(size=size)
    worker_va = subprocess.Popen(cmd, shell=True) 

    worker_tr.communicate()
    worker_va.communicate()

    cmd = './fm-train -u 3 -q -s 192 -t 40 va.r{size}.fm2 tr.r{size}.fm2'.format(size=size) 
    subprocess.call(cmd, shell=True) 

print('time used = {0}'.format(time.time()-start))
