#!/usr/bin/env python3

import threading, subprocess, uuid, random, queue, argparse, sys, math, os

class Worker(threading.Thread):
    def __init__(self, cvt_path, src_path, dst_path, lock):
        threading.Thread.__init__(self)
        self.cvt_path = cvt_path
        self.src_path = src_path
        self.dst_path = dst_path
        self.lock = lock
        
    def run(self):
        cmd = '{0} {1} {2}'.\
            format(os.path.join('.', self.cvt_path), self.src_path, self.dst_path)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE)
        p.communicate()

def split(path, nr_threads):

    def open_skip_first_line(path):
        f = open(path)
        next(f)
        return f

    def open_with_header_witten(path):
        f = open(path+'.__tmp__.{0}'.format(idx), 'w')
        f.write('dummy\n')
        return f

    def calc_nr_lines_per_thread():
        nr_lines = int(list(subprocess.Popen('wc -l {0}'.format(path), shell=True, 
            stdout=subprocess.PIPE).stdout)[0].split()[0])
        return math.ceil(float(nr_lines-1)/nr_threads)

    nr_lines_per_thread = calc_nr_lines_per_thread()

    idx = 0
    f = open_with_header_witten(path)
    for i, line in enumerate(open_skip_first_line(path), start=1):
        if i%nr_lines_per_thread == 0:
            f.close()
            idx += 1
            f = open_with_header_witten(path)
        f.write(line)
    f.close()

def parse_args():
    
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    parser = argparse.ArgumentParser(description='parallel convert')
    parser.add_argument('-n', default=12, type=int, help='set number of path')
    parser.add_argument('cvt_path', type=str, help='set the path to your desired converter')
    parser.add_argument('csv_path', type=str, help='set path to the csv file')
    parser.add_argument('svm_path', type=str, help='set path to the svm file')
    args = vars(parser.parse_args())

    return args

def parallel_convert(args):

    workers, lock = [], threading.Lock()
    for i in range(args['n']):
        worker = Worker(
            args['cvt_path'], 
            args['csv_path']+'.__tmp__.{0}'.format(i),
            args['svm_path']+'.__tmp__.{0}'.format(i),
            lock)
        workers.append(worker)

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

def cat_svm_files(args):
    
    if os.path.exists(args['svm_path']):
        os.remove(args['svm_path'])
    for i in range(args['n']):
        cmd = 'cat {svm}.__tmp__.{idx} >> {svm}'.format(svm=args['svm_path'], idx=i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

def del_files(args):
    
    for i in range(args['n']):
        os.remove('{0}.__tmp__.{1}'.format(args['csv_path'], i))
        os.remove('{0}.__tmp__.{1}'.format(args['svm_path'], i))

def main():
    
    args = parse_args()

    split(args['csv_path'], args['n'])

    parallel_convert(args)

    cat_svm_files(args)

    del_files(args)

main()
