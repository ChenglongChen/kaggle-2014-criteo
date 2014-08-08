#!/usr/bin/env python3

import subprocess, sys, multiprocessing, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int, default=100)
args = vars(parser.parse_args())

def run(c, queue):
    cmd = './supertrain -s 7 -c {0} tr.r{1}.svm model.{0} && ./superpredict va.r{1}.svm model.{0} /dev/null'.format(c, args['size'])
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    for line in output:
        line = line.decode('utf-8')
        if line.startswith('log loss'):
            logloss = float(line.strip().split('=')[1])
    queue.put((c,logloss))

queue, workers = multiprocessing.Queue(), []
for c in [2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]:
    workers.append(multiprocessing.Process(target=run, args=(c, queue)))

for worker in workers:
    worker.start()

for worker in workers:
    worker.join()

best_c, best_loss = -1, 1000
while not queue.empty():
    c, logloss = queue.get()
    if logloss < best_loss:
        best_c, best_loss = c, logloss
print('{0:.2f} {1}'.format(best_c, best_loss))
