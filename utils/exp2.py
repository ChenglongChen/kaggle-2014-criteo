#!/usr/bin/env python3

import subprocess, sys, os, time

def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    subprocess.call('renice -n 10 -u r01922136', shell=True, stdout=subprocess.PIPE)
    p.communicate()

#cmd = 'git clean -df && utils/prepare.sh'
#subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start = time.time()
for size in ["x"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer3.py -s 24 converters/num3.py tr.r{size}.csv tr.r{size}.num.dense tr.r{size}.num.sparse'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer3.py -s 24 converters/num3.py va.r{size}.csv va.r{size}.num.dense va.r{size}.num.sparse'.format(size=size)
    run(cmd) 

    cmd = './mark44 -t 30 -s 24 va.r{size}.num.dense va.r{size}.num.sparse tr.r{size}.num.dense tr.r{size}.num.sparse va.r{size}.num.out tr.r{size}.num.out'.format(size=size) 
    run(cmd)

    cmd = 'converters/parallelizer2.py -s 24 converters/combine.py tr.r{size}.csv tr.r{size}.num.out tr.r{size}.fm'.format(size=size)
    #run(cmd) 
    cmd = 'converters/parallelizer2.py -s 24 converters/combine.py va.r{size}.csv va.r{size}.num.out va.r{size}.fm'.format(size=size)
    #run(cmd) 

    cmd = './mark33 -s 24 -t 15 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    #run(cmd) 

print('time used = {0:.0f}'.format(time.time()-start))
