#!/usr/bin/env python3

import subprocess, sys, os, time

def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    subprocess.call('renice -n 5 -u r01922136', shell=True, stdout=subprocess.PIPE)
    p.communicate()

cmd = 'git clean -df && utils/prepare.sh && ./converters/dump.py'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["x", "1"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer.py -n 36 converters/num.py tr.r{size}.csv tr.r{size}.gbdt'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer.py -n 36 converters/num.py va.r{size}.csv va.r{size}.gbdt'.format(size=size)
    run(cmd) 

    cmd = './mark29 -t 30 -s 13 va.r{size}.gbdt tr.r{size}.gbdt'.format(size=size) 
    run(cmd)

    cmd = 'converters/parallelizer.py -n 36 converters/gbdt.py tr.r{size}.csv tr.r{size}.gbdt.out tr.r{size}.fm'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer.py -n 36 converters/gbdt.py va.r{size}.csv va.r{size}.gbdt.out va.r{size}.fm'.format(size=size)
    run(cmd) 

    cmd = './mark33 -s 36 -t 30 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    run(cmd) 

print('time used = {0:.0f}'.format(time.time()-start))
