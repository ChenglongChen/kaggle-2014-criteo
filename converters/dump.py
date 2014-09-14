#!/usr/bin/env python3

import argparse, csv, sys, pickle

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

frequent_feats = read_freqent_feats()

pickle.dump(frequent_feats, open('fc.trva.r1.p1.t10.pickle', 'wb'))
