#!/usr/bin/env python3

import threading, subprocess, uuid, random, queue, argparse, sys, math, os

from common import *

def split(path, nr_threads):

    def open_with_header_witten(path):
        f = open(path+'.__tmp__.{0}'.format(idx), 'w')
        f.write('Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26\n')
        return f

    def calc_nr_lines_per_thread():
        nr_lines = int(list(subprocess.Popen('wc -l {0}'.format(path), shell=True, 
            stdout=subprocess.PIPE).stdout)[0].split()[0])
        return math.ceil(float(nr_lines+1)/nr_threads)

    nr_lines_per_thread = calc_nr_lines_per_thread()

    idx = 0
    f = open_with_header_witten(path)
    for i, line in enumerate(open_with_first_line_skipped(path), start=1):
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
    parser.add_argument('-n', default=12, type=int, help='set number of threads')
    parser.add_argument('cvt_path', type=str, help='set the path to your desired converter')
    parser.add_argument('csv_path', type=str, help='set path to the csv file')
    parser.add_argument('svm_path', type=str, help='set path to the svm file')
    args = vars(parser.parse_args())

    return args

def parallel_convert(args):

    workers, lock = [], threading.Lock()
    for i in range(args['n']):
        cmd = '{0} {1} {2}'.format(
            os.path.join('.', args['cvt_path']),
            args['csv_path']+'.__tmp__.{0}'.format(i),
            args['svm_path']+'.__tmp__.{0}'.format(i))
        worker = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        workers.append(worker)
    for worker in workers:
        worker.communicate()

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
