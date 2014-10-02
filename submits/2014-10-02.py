#!/usr/bin/env python3

import subprocess, sys, os, time

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

def run(cmd, stdout=None):
    p = subprocess.Popen(cmd, shell=True, stdout=stdout)
    #subprocess.call('renice -n 10 -u r01922136', shell=True, stdout=subprocess.PIPE)
    p.communicate()

cmd = 'git add {me} && git clean -df && utils/prepare.sh'.format(me=sys.argv[0])
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

start = time.time()
for size in ["x", "0"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer1.py -s 24 converters/num.py tr.r{size}.csv tr.r{size}.num.dense tr.r{size}.num.sparse'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer1.py -s 24 converters/num.py va.r{size}.csv va.r{size}.num.dense va.r{size}.num.sparse'.format(size=size)
    run(cmd) 

    print('prepressing 1 time = {0:.0f}'.format(time.time()-start))

    cmd = './mark48 -t 30 -s 24 va.r{size}.num.dense va.r{size}.num.sparse tr.r{size}.num.dense tr.r{size}.num.sparse va.r{size}.num.out tr.r{size}.num.out'.format(size=size) 
    run(cmd)

    print('gbdt 1 time = {0:.0f}'.format(time.time()-start))

    cmd = 'rm -f va.r{size}.num.dense va.r{size}.num.sparse tr.r{size}.num.dense tr.r{size}.num.sparse'.format(size=size) 
    run(cmd)

    cmd = 'converters/parallelizer2.py -s 24 converters/combine.py tr.r{size}.csv tr.r{size}.num.out tr.r{size}.fm'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer2.py -s 24 converters/combine.py va.r{size}.csv va.r{size}.num.out va.r{size}.fm'.format(size=size)
    run(cmd) 

    cmd = 'rm -f va.r{size}.num.out va.r{size}.num.out'.format(size=size) 
    run(cmd)

    print('preprosessing 2 time = {0:.0f}'.format(time.time()-start))

    cmd = './mark33 -k 4 -s 24 -t 11 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    run(cmd) 

    print('FM time = {0:.0f}'.format(time.time()-start))

print('time used = {0:.0f}'.format(time.time()-start))

cmd = 'utils/make_submission.py va.r0.fm.out {uuid}.csv && git add {log_dir} {me} && git commit --allow-empty-message -m "" && git tag {uuid}'\
    .format(uuid=UUID, log_dir=LOG_DIR, me=sys.argv[0])
subprocess.call(cmd, shell=True)
print('remember to run "git push && git push --tags"')
