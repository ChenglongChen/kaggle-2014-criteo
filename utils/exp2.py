#!/usr/bin/env python3

import subprocess, sys, os, time

def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    subprocess.call('renice -n 10 -u r01922136', shell=True, stdout=subprocess.PIPE)
    p.communicate()

cmd = 'git clean -df && utils/prepare.sh && ./converters/dump.py'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["10"]:
    for c in range(1, 13):
        print('size = {size}, c = {c}'.format(size=size, c=c))

        cmd = 'converters/parallelizer.py -n 24 "converters/num2.py -c {c}" tr.r{size}.csv tr.r{size}.num'.format(size=size, c=c)
        run(cmd) 
        cmd = 'converters/parallelizer.py -n 24 "converters/num2.py -c {c}" va.r{size}.csv va.r{size}.num'.format(size=size, c=c)
        run(cmd) 

        cmd = './mark36 -t 30 -s 13 va.r{size}.num tr.r{size}.num'.format(size=size) 
        run(cmd)

        cmd = 'converters/parallelizer.py -n 24 converters/combine.py tr.r{size}.csv tr.r{size}.num.out tr.r{size}.fm'.format(size=size)
        run(cmd) 
        cmd = 'converters/parallelizer.py -n 24 converters/combine.py va.r{size}.csv va.r{size}.num.out va.r{size}.fm'.format(size=size)
        run(cmd) 

        cmd = './mark33 -s 24 -t 15 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
        run(cmd) 

print('time used = {0:.0f}'.format(time.time()-start))
