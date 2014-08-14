#!/usr/bin/env python3

import subprocess, sys, os

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

cmd = 'git clean -df && utils/prepare.sh'
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

workers = []
for size in ["x", "100", "10", "1"]:
    for data in ['tr', 'va']:
        cmd = 'converters/parallelizer.py -n 24 converters/defender.py {data}.r{size}.csv {data}.r{size}.svm'\
            .format(size=size, data=data)
        subprocess.call(cmd, shell=True) 
    cmd = './sgd-poly2-train-fast -t 5 -v va.r{size}.svm tr.r{size}.svm model.r{size} && '.format(size=size, log_dir=LOG_DIR) 
    cmd += './sgd-poly2-predict-fast va.r{size}.svm model.r{size} out.r{size}'.format(size=size, log_dir=LOG_DIR)
    log_path = '{log_dir}/log.r{size}'.format(size=size, log_dir=LOG_DIR)
    worker = subprocess.Popen(cmd, shell=True, stdout=open(log_path, 'w')) 
    workers.append(worker)

for worker in workers:
    worker.communicate()

if UUID != 'exp':
    cmd = 'git add {log_dir} && git commit --allow-empty-message -m ""'.format(uuid=UUID, log_dir=LOG_DIR)
    subprocess.call(cmd, shell=True)
