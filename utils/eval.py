#!/usr/bin/env python

import sys
import numpy as np

from sklearn import metrics


def usage():
    print 'Validation Set, Prediction Output'
    exit(1)


if len(sys.argv) != 3:
    print usage()

va_path = sys.argv[1]
pred_path = sys.argv[2]

va_list = []
pred_list = []

for idx, (va_line, pred_line) in enumerate(zip(open(va_path), open(pred_path))):
    va_list.append(float(va_line.split()[0]))
    pred_list.append(float(pred_line.split()[0]))

va_array = np.array(va_list)
pred_array = np.array(pred_list)

# Computer AUC
fpr, tpr, thresholds = metrics.roc_curve(va_array, pred_array)
AUC = metrics.auc(fpr, tpr)
print 'AUC:\t', AUC
