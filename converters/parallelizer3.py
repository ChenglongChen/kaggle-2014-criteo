#!/usr/bin/env python3

import threading, subprocess, uuid, random, queue, argparse, sys, math, os

from common import *

def split(path, args):

    def open_with_header_witten(path):
        f = open(path+'.__tmp__.{0}'.format(idx), 'w')
        if args['ignore_header']:
            return f 
        f.write(header)
        return f

    def calc_nr_lines_per_thread():
        nr_lines = int(list(subprocess.Popen('wc -l {0}'.format(path), shell=True, 
            stdout=subprocess.PIPE).stdout)[0].split()[0])
        return math.ceil(float(nr_lines+1)/args['nr_thread'])

    def get_header(path):
        return open(path).readline()

    nr_lines_per_thread = calc_nr_lines_per_thread()

    idx = 0
    header = get_header(path)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='nr_thread', default=12, type=int)
    parser.add_argument('cvt_path')
    parser.add_argument('src_path')
    parser.add_argument('dst_path')
    parser.add_argument('dst2_path')
    args = vars(parser.parse_args())

    return args

def parallel_convert(paths, nr_threads):

    workers, lock = [], threading.Lock()
    for i in range(args['nr_thread']):
        cmd = '{0}'.format(os.path.join('.', args['cvt_path']))
        for path in paths:
            cmd += ' {0}'.format(path+'.__tmp__.{0}'.format(i))
        cmd += ' {0}'.format(args['dst_path']+'.__tmp__.{0}'.format(i))
        worker = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        workers.append(worker)
    for worker in workers:
        worker.communicate()

def cat_dst_files(path, nr_threads):
    
    if os.path.exists(path):
        os.remove(path)
    for i in range(nr_threads):
        cmd = 'cat {svm}.__tmp__.{idx} >> {svm}'.format(svm=path, idx=i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

def del_files(args):
    
    for i in range(args['nr_thread']):
        os.remove('{0}.__tmp__.{1}'.format(args['src_path'], i))
        os.remove('{0}.__tmp__.{1}'.format(args['dst_path'], i))

def main():
    
    args = parse_args()

    split(args['src_path'], args)

    parallel_convert(args)

    cat_dst_files(args['dst_path'], args['nr_thread'])

    cat_dst_files(args['dst2_path'], args['nr_thread'])

    del_files(args)

main()
