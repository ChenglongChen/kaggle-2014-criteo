#!/usr/bin/env python3

import subprocess, os

def check(path, target_md5sum):
    p = subprocess.Popen('md5sum {0}'.format(path).split(), stdout=subprocess.PIPE)
    stdout = p.stdout.readline().decode('utf-8')
    md5sum = stdout.split()[0]
    if md5sum == target_md5sum:
        return True
    else:
        return False

success = True
if not os.path.exists('train.csv'):
    print('train.csv does not exist')
    success = False 
elif not check('train.csv', 'ebf87fe3daa7d729e5c9302947050f41'):
    print('train.csv is incorrect')
    success = False 

if not os.path.exists('test.csv'):
    print('test.csv does not exist')
    success = False 
elif not check('te.rx.csv', '8016f59e45abb37ae7f6e7956f30e052'):
    print('test.csv is incorrect')
    success = False 

if success:
    print('Your csv files are correct!')
