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
    tr_csv = 'tr.r{size}.csv'.format(size=size)
    tr_fm = 'tr.r{size}.fm'.format(size=size)
    va_csv = 'va.r{size}.csv'.format(size=size)
    va_fm = 'va.r{size}.fm'.format(size=size)
    model = 'model.r{size}'.format(size=size)
    out = 'r{size}.out'.format(size=size)
    log_file = open('{log_dir}/log.r{size}'.format(log_dir=LOG_DIR, size=size), 'w')

    for data_csv, data_fm in [(tr_csv, tr_fm), (va_csv, va_fm)]:
        cmd = 'converters/parallelizer.py -n 24 converters/defender.py {data_csv} {data_fm}'\
            .format(data_csv=data_csv, data_fm=data_fm)
        subprocess.call(cmd, shell=True)

    cmd = './fm-train -q -s 24 -t 15 -v {va_fm} {tr_fm} {model}'.format(va_fm=va_fm, tr_fm=tr_fm, model=model) 
    #cmd += ' && ./fm-predict {va_fm} {model} {out}'.format(va_fm=va_fm, model=model, out=out)
    #cmd += ' && ./utils/calc_log_loss.py {va_fm} {out}'.format(va_fm=va_fm, out=out)
    worker = subprocess.Popen(cmd, shell=True, stdout=log_file) 
    worker.communicate()
    log_file.close()
print('time used = {0}'.format(time.time()-start))

if UUID != 'exp':
    cmd = 'git add {log_dir} && git commit --allow-empty-message -m ""'.format(uuid=UUID, log_dir=LOG_DIR)
    subprocess.call(cmd, shell=True)
