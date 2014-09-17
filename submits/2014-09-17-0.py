#!/usr/bin/env python3

import subprocess, sys, os, time

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    subprocess.call('renice -n 10 -u r01922136', shell=True, stdout=subprocess.PIPE)
    p.communicate()

cmd = 'git add {me} && git clean -df && utils/prepare.sh'.format(me=sys.argv[0])
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

start = time.time()
for size in ["x", "1", "0"]:
    print('size = {size}'.format(size=size))

    cmd = 'converters/parallelizer.py -s 24 converters/num2.py tr.r{size}.csv tr.r{size}.num'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer.py -s 24 converters/num2.py va.r{size}.csv va.r{size}.num'.format(size=size)
    run(cmd) 

    cmd = './mark29 -t 30 -s 24 va.r{size}.num tr.r{size}.num'.format(size=size) 
    run(cmd)

    cmd = 'converters/parallelizer.py -s 24 converters/combine.py tr.r{size}.csv tr.r{size}.num.out tr.r{size}.fm'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer.py -s 24 converters/combine.py va.r{size}.csv va.r{size}.num.out va.r{size}.fm'.format(size=size)
    run(cmd) 

    cmd = './mark33 -s 24 -t 11 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    run(cmd) 

print('time used = {0:.0f}'.format(time.time()-start))

cmd = 'utils/make_submission.py va.r0.fm.out {uuid}.csv && git add {log_dir} {me} && git commit --allow-empty-message -m "" && git tag {uuid}'\
    .format(uuid=UUID, log_dir=LOG_DIR, me=sys.argv[0])
subprocess.call(cmd, shell=True)
print('remember to run "git push && git push --tags"')
