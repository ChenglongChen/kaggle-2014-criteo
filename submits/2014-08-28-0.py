#!/usr/bin/env python3

import subprocess, sys, os, time

UUID = os.path.splitext(os.path.basename(sys.argv[0]))[0]
LOG_DIR = 'logs/{0}'.format(UUID)

cmd = 'git add {me} && git clean -df && utils/prepare.sh'.format(me=sys.argv[0])
subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

start = time.time()
for size in ["100", "10", "1", "0"]:
    tr_csv = 'tr.r{size}.csv'.format(size=size)
    tr_svm = 'tr.r{size}.svm'.format(size=size)
    va_csv = 'va.r{size}.csv'.format(size=size)
    va_svm = 'va.r{size}.svm'.format(size=size)
    model = 'model.r{size}'.format(size=size)
    out = 'out.r{size}'.format(size=size)
    log_path = '{log_dir}/log.r{size}'.format(log_dir=LOG_DIR, size=size)

    for data_csv, data_svm in [(tr_csv, tr_svm), (va_csv, va_svm)]:
        cmd = 'converters/parallelizer.py -n 24 converters/challenger.py {data_csv} {data_svm}'\
            .format(data_csv=data_csv, data_svm=data_svm)
        subprocess.call(cmd, shell=True)

    cmd = './fm-train -s 24 -t 10 -v {va_svm} {tr_svm} {model}'.format(va_svm=va_svm, tr_svm=tr_svm, model=model) 
    cmd += ' && ./fm-predict {tr_svm} {model} tr.{out}'.format(tr_svm=tr_svm, model=model, out=out)
    cmd += ' && ./fm-predict {va_svm} {model} {out}'.format(va_svm=va_svm, model=model, out=out)
    cmd += ' && ./utils/calc_log_loss.py {va_svm} {out}'.format(va_svm=va_svm, out=out)
    worker = subprocess.Popen(cmd, shell=True, stdout=open(log_path, 'w')) 
    worker.communicate()
print('time used = {0}'.format(time.time()-start))

cmd = 'utils/make_submission.py out.r0 {uuid}.csv && git add {log_dir} {me} && git commit --allow-empty-message -m "" && git tag {uuid}'\
    .format(uuid=UUID, log_dir=LOG_DIR, me=sys.argv[0])
subprocess.call(cmd, shell=True)
print('remember to run "git push && git push --tags"')
