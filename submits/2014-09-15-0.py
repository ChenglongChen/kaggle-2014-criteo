#!/usr/bin/env python3

import subprocess, sys, os, time

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

cmd = 'git add {me} && git clean -df && utils/prepare.sh && ./converters/dump.py'.format(me=sys.argv[0])
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

start = time.time()
for size in ["x", "100", "10", "1", "0"]:
    print('size = {size}'.format(size=size))

    f_log = open('{log_dir}/log.r{size}'.format(log_dir=LOG_DIR, size=size), 'w')

    cmd = 'converters/parallelizer.py -n 96 converters/num.py tr.r{size}.csv tr.r{size}.gbdt'.format(size=size)
    subprocess.call(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 96 converters/num.py va.r{size}.csv va.r{size}.gbdt'.format(size=size)
    subprocess.call(cmd, shell=True) 

    cmd = './mark29 -t 30 -s 13 va.r{size}.gbdt tr.r{size}.gbdt'.format(size=size) 
    subprocess.call(cmd, shell=True, stdout=f_log)

    cmd = 'converters/parallelizer.py -n 96 converters/gbdt.py tr.r{size}.csv tr.r{size}.gbdt.out tr.r{size}.fm'.format(size=size)
    subprocess.call(cmd, shell=True) 
    cmd = 'converters/parallelizer.py -n 96 converters/gbdt.py va.r{size}.csv va.r{size}.gbdt.out va.r{size}.fm'.format(size=size)
    subprocess.call(cmd, shell=True) 

    cmd = './mark33 -s 220 -t 20 va.r{size}.fm tr.r{size}.fm'.format(size=size) 
    subprocess.call(cmd, shell=True, stdout=f_log) 

print('time used = {0:.0f}'.format(time.time()-start))

cmd = 'utils/make_submission.py va.r0.fm.out {uuid}.csv && git add {log_dir} {me} && git commit --allow-empty-message -m "" && git tag {uuid}'\
    .format(uuid=UUID, log_dir=LOG_DIR, me=sys.argv[0])
subprocess.call(cmd, shell=True)
print('remember to run "git push && git push --tags"')
