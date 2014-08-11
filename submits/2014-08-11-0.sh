#!/bin/bash

TR="trva"
VA="te"

source ~/.bashrc
gcndf
./utils/prepare.sh
converters/parallel_convert.py -n 24 "converters/converter-defender.py" "$TR"".csv" "$TR"".svm"
converters/parallel_convert.py -n 24 "converters/converter-defender.py" "$VA"".csv" "$VA"".svm"
./sgd-poly2-train -t 5 "$TR"".svm" model
./sgd-poly2-predict "$VA"".svm" model out.txt
utils/make_submission.py out.txt "2014-08-11-0.csv"
