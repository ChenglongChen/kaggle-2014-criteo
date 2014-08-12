#!/usr/bin/env python3

import subprocess, sys, os

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

cmd = 'git add {me} && git clean -df && utils/prepare.sh'.format(me=sys.argv[0])
subprocess.call(cmd, shell=True)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

workers = []
for size in [100, 10, 1, 0]:
    cmd = 'converters/parallelizer.py -n 24 converters/challenger.py tr.r{size}.csv tr.r{size}.svm'.format(size=size)
    subprocess.call(cmd, shell=True) 
    cmd = cmd.replace('tr', 'va')
    subprocess.call(cmd, shell=True) 
    cmd = './sgd-poly2-train-fast -t 5 -v va.r{size}.svm tr.r{size}.svm model.r{size} > {log_dir}/log.r{size} && '.format(size=size, log_dir=LOG_DIR) 
    cmd += './sgd-poly2-predict-fast va.r{size}.svm model.r{size} out.r{size} >> {log_dir}/log.r{size}'.format(size=size, log_dir=LOG_DIR)
    worker = subprocess.Popen(cmd, shell=True) 
    workers.append(worker)

for worker in workers:
    worker.communicate()

cmd = 'utils/make_submission.py out.r0 {uuid}.csv && git add {log_dir} {me} && git commit --allow-empty-message -m "" && git tag {uuid}'.format(uuid=UUID, log_dir=LOG_DIR, me=sys.argv[0])
subprocess.call(cmd, shell=True)
