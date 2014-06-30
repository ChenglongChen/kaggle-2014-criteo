#!/usr/bin/env python3


import threading, subprocess, uuid, random, queue

WORKERS = ['linux7']
MAX_WORKERS = 1

class Worker(threading.Thread):
    def __init__(self, host, task_queue, best, lock):
        threading.Thread.__init__(self)
        self.host = host
        self.task_queue = task_queue
        self.best = best
        self.lock = lock
        
    def run(self):
        while True:
            spec = self.task_queue.get()
            if spec is None:
                self.task_queue.put(None)
                break
            try:
                self.run_one(spec)
            except Exception as e:
                print(e)
                self.task_queue.put(spec)
                print('{0} quit.\n'.format(self.host))
                break

    def run_one(self, spec):
        cmd = "{0}".format(self.get_cmd(spec))
        print(cmd)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        p.communicate()
        self.lock.acquire()
        #if logloss < self.best['logloss']:
        #    self.best['logloss'] = logloss
        #    self.best['field'] = spec['field']
        #    output += ' *'
        print('{0} done'.format(spec['field']))
        self.lock.release()

    def get_cmd(self, spec):
        cmd = './convert_to_svm.v2.py -i {0} -- tr.1m.csv tr.1m.v4.i{0}.svm;'.format(spec['field'])
        cmd += './convert_to_svm.v2.py -i {0} -- va.100k.csv va.100k.v4.i{0}.svm'.format(spec['field'])
        return cmd

    def __del__(self):
        p = subprocess.Popen('rm -rf __tmp__*'.format(self.host), 
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()

def main():

    best = {'logloss': float('Inf'), 'field': -1}

    task_queue = queue.Queue()
    for field in range(1, 14):
        task_queue.put({'field':field})
    task_queue.put(None)

    lock = threading.Lock()
    workers = [Worker(random.choice(WORKERS), task_queue, best, lock)
        for i in range(min(len(WORKERS)*MAX_WORKERS, task_queue.qsize()))]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    print(best)

main()
