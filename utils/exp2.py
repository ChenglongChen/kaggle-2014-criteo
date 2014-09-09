#!/usr/bin/env python3

import subprocess, sys, os, time

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

cmd = 'git clean -df && utils/prepare.sh'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

start = time.time()
for size in ["100", "10", "1"]:
    log_file = open('{log_dir}/log.r{size}'.format(log_dir=LOG_DIR, size=size), 'w')

    for data in ['tr', 'va']:
        cmd = 'converters/parallelizer.py -n 24 converters/num.py {data}.r{size}.csv {data}.r{size}.svm.num'.format(size=size, data=data)
        subprocess.call(cmd, shell=True)

    for data in ['tr', 'va']:
        cmd = 'converters/parallelizer.py -n 24 converters/defender.py {data}.r{size}.csv {data}.r{size}.fm'.format(size=size, data=data)
        subprocess.call(cmd, shell=True)

    cmd = './mark29 -q -s 24 -v va.r{size}.svm.num tr.r{size}.svm.num'.format(size=size) 
    subprocess.call(cmd, shell=True)

    for data in ['tr', 'va']:
        cmd = 'converters/parallelizer.py -n 24 converters/combine.py {data}.r{size}.fm {data}.r{size}.svm.num.gbdt {data}.r{size}.fm2'.format(size=size, data=data)
        subprocess.call(cmd, shell=True)

    cmd = './fm-train -q -s 24 -t 20 -v va.r{size}.fm2 tr.r{size}.fm2'.format(size=size) 
    subprocess.call(cmd, shell=True, stdout=log_file) 

    log_file.close()

print('time used = {0}'.format(time.time()-start))

if UUID != 'exp':
    cmd = 'git add {log_dir} && git commit --allow-empty-message -m ""'.format(uuid=UUID, log_dir=LOG_DIR)
    subprocess.call(cmd, shell=True)
