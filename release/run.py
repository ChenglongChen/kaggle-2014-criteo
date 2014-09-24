#!/usr/bin/env python3

import subprocess, sys, os, time

def run(cmd):
    subprocess.call(cmd, shell=True)

cmd = 'utils/prepare.sh'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["x"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer.py -s 24 converters/num.py tr.r{size}.csv tr.r{size}.num'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer.py -s 24 converters/num.py va.r{size}.csv va.r{size}.num'.format(size=size)
    run(cmd) 

    cmd = './mark29 -t 30 -s 24 va.r{size}.num tr.r{size}.num'.format(size=size) 
    run(cmd)

    cmd = 'converters/parallelizer2.py -s 24 converters/combine.py tr.r{size}.csv tr.r{size}.num.out tr.r{size}.fm'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer2.py -s 24 converters/combine.py va.r{size}.csv va.r{size}.num.out va.r{size}.fm'.format(size=size)
    run(cmd) 

    cmd = './mark33 -s 24 -t 15 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    run(cmd) 

print('time used = {0:.0f}'.format(time.time()-start))
