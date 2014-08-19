#!/usr/bin/env python3

import subprocess, sys, os, time, itertools

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

#cmd = 'git clean -df && utils/prepare.sh'
#subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#
#if not os.path.exists(LOG_DIR):
#    os.makedirs(LOG_DIR)

start = time.time()
for size in ["100"]:
    tr_csv = 'tr.r{size}.csv'.format(size=size)
    tr_svm = 'tr.r{size}.svm'.format(size=size)
    va_csv = 'va.r{size}.csv'.format(size=size)
    va_svm = 'va.r{size}.svm'.format(size=size)
    model = 'model.r{size}'.format(size=size)
    out = 'out.r{size}'.format(size=size)
    log_path = 'x.log'
    worker = subprocess.Popen('echo "" > {0}'.format(log_path), shell=True) 
    worker.communicate()

    for feat1, feat2 in itertools.combinations(range(1, 39), 2):
        for data_csv, data_svm in [(tr_csv, tr_svm), (va_csv, va_svm)]:
            cmd = 'converters/parallelizer.py -n 24 "converters/challenger-1.py -f {feat1},{feat2}" {data_csv} {data_svm}'\
                .format(data_csv=data_csv, data_svm=data_svm, feat1=feat1, feat2=feat2)
            subprocess.call(cmd, shell=True)

        cmd = './fm-sse-train -l 0.01 -q -s 24 -t 10 -v {va_svm} {tr_svm} {model}'.format(va_svm=va_svm, tr_svm=tr_svm, model=model) 
        #cmd += ' && ./fm-sse-predict {va_svm} {model} {out}'.format(va_svm=va_svm, model=model, out=out)
        #cmd += ' && ./utils/calc_log_loss.py {va_svm} {out}'.format(va_svm=va_svm, out=out)
        print('f1 = {feat1}, f2 = {feat2}'.format(feat1=feat1, feat2=feat2)) 
        worker = subprocess.Popen('echo "f1 = {feat1}, f2 = {feat2}" >> {log_path}'.format(feat1=feat1, feat2=feat2, log_path=log_path), shell=True) 
        worker.communicate()
        worker = subprocess.Popen(cmd, shell=True, stdout=open(log_path, 'a')) 
        worker.communicate()
print('time used = {0}'.format(time.time()-start))

#if UUID != 'exp':
#    cmd = 'git add {log_dir} && git commit --allow-empty-message -m ""'.format(uuid=UUID, log_dir=LOG_DIR)
#    subprocess.call(cmd, shell=True)