#!/usr/bin/env python3

import subprocess, sys, os, time

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

def run(cmd, stdout=None):
    p = subprocess.Popen(cmd, shell=True, stdout=stdout)
    subprocess.call('renice -n 10 -u r01922136', shell=True, stdout=subprocess.PIPE)
    p.communicate()

cmd = 'git add {me} && git clean -df && utils/prepare.sh'.format(me=sys.argv[0])
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

start = time.time()
for size in ["x", "0"]:
    print('size = {size}'.format(size=size))

    f_log = open('{log_dir}/log.r{size}'.format(log_dir=LOG_DIR, size=size), 'w')

    cmd = 'converters/parallelizer.py -s 24 converters/num2.py tr.r{size}.csv tr.r{size}.num'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer.py -s 24 converters/num2.py va.r{size}.csv va.r{size}.num'.format(size=size)
    run(cmd) 

    cmd = './mark29 -t 50 -s 24 va.r{size}.num tr.r{size}.num'.format(size=size) 
    run(cmd, f_log)

    cmd = 'converters/parallelizer2.py -s 24 converters/combine.py tr.r{size}.csv tr.r{size}.num.out tr.r{size}.fm'.format(size=size)
    run(cmd) 
    cmd = 'converters/parallelizer2.py -s 24 converters/combine.py va.r{size}.csv va.r{size}.num.out va.r{size}.fm'.format(size=size)
    run(cmd) 

    cmd = './mark33 -k 8 -s 24 -t 17 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    run(cmd, f_log) 

    f_log.close()

print('time used = {0:.0f}'.format(time.time()-start))

cmd = 'utils/make_submission.py va.r0.fm.out {uuid}.csv && git add {log_dir} {me} && git commit --allow-empty-message -m "" && git tag {uuid}'\
    .format(uuid=UUID, log_dir=LOG_DIR, me=sys.argv[0])
subprocess.call(cmd, shell=True)
print('remember to run "git push && git push --tags"')
