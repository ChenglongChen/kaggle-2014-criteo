#!/usr/bin/env python3

import subprocess, sys, os, time

def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    subprocess.call('renice -n 10 -u r01922136', shell=True, stdout=subprocess.PIPE)
    p.communicate()

cmd = 'git clean -df && utils/prepare.sh'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["x", "1"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer-a.py -s 24 converters/pre-a.py tr.r{size}.csv tr.r{size}.gbdt.dense tr.r{size}.gbdt.sparse'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer-a.py -s 24 converters/pre-a.py va.r{size}.csv va.r{size}.gbdt.dense va.r{size}.gbdt.sparse'.format(size=size)
    run(cmd) 

    cmd = './gbdt -t 30 -s 24 va.r{size}.gbdt.dense va.r{size}.gbdt.sparse tr.r{size}.gbdt.dense tr.r{size}.gbdt.sparse va.r{size}.gbdt.out tr.r{size}.gbdt.out'.format(size=size) 
    run(cmd)

    cmd = 'converters/parallelizer-b.py -s 24 converters/pre-b.py tr.r{size}.csv tr.r{size}.gbdt.out tr.r{size}.fm'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer-b.py -s 24 converters/pre-b.py va.r{size}.csv va.r{size}.gbdt.out va.r{size}.fm'.format(size=size)
    run(cmd) 

    cmd = './fm -s 24 -t 11 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    run(cmd) 

print('time used = {0:.0f}'.format(time.time()-start))
