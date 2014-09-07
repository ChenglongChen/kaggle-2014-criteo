#!/bin/bash

if [[ "$(hostname)" == linux* ]]; then
cd ~/work2/kaggle.2014.criteo/
elif [[ "$(hostname)" == optima ]] || [[ "$(hostname)" == schroeder ]]; then
cd ~/work/kaggle.2014.criteo/
fi
#make -C solvers/superliblinear/
#ln -sf solvers/superliblinear/train supertrain
#ln -sf solvers/superliblinear/predict superpredict
#make -C solvers/sgd-poly2/
#ln -sf solvers/sgd-poly2/sgd-poly2-train .
#ln -sf solvers/sgd-poly2/sgd-poly2-predict .
#make -C solvers/sgd-poly2-fast/
#ln -sf solvers/sgd-poly2-fast/sgd-poly2-train-fast .
#ln -sf solvers/sgd-poly2-fast/sgd-poly2-predict-fast .
#make -C solvers/sgd-poly3-fast/
#ln -sf solvers/sgd-poly3-fast/sgd-poly3-train-fast .
#ln -sf solvers/sgd-poly3-fast/sgd-poly3-predict-fast .
#make -C solvers/sgd-poly2-fast-l1/
#ln -sf solvers/sgd-poly2-fast-l1/sgd-poly2-train-fast-l1 .
#ln -sf solvers/sgd-poly2-fast-l1/sgd-poly2-predict-fast-l1 .
#make -C solvers/sgd-poly2-fast-sp0/
#ln -sf solvers/sgd-poly2-fast-sp0/sgd-poly2-train-fast-sp0 .
#ln -sf solvers/sgd-poly2-fast-sp0/sgd-poly2-predict-fast-sp0 .
#make -C solvers/sgd-poly2-fast-sp1/
#ln -sf solvers/sgd-poly2-fast-sp1/sgd-poly2-train-fast-sp1 .
#ln -sf solvers/sgd-poly2-fast-sp1/sgd-poly2-predict-fast-sp1 .
#make -C solvers/sgd-poly2-fast-sp2/
#ln -sf solvers/sgd-poly2-fast-sp2/sgd-poly2-train-fast-sp2 .
#ln -sf solvers/sgd-poly2-fast-sp2/sgd-poly2-predict-fast-sp2 .
#make -C solvers/sgd-poly2-fast-sp3/
#ln -sf solvers/sgd-poly2-fast-sp3/sgd-poly2-train-fast-sp3 .
#ln -sf solvers/sgd-poly2-fast-sp3/sgd-poly2-predict-fast-sp3 .
#make -C solvers/superliblinear-poly2/
#ln -sf solvers/superliblinear-poly2/train supertrain-poly2
#ln -sf solvers/superliblinear-poly2/predict superpredict-poly2
#make -C solvers/fm
#ln -sf solvers/fm/fm-train .
#ln -sf solvers/fm/fm-predict .
make -C solvers/fm
ln -sf solvers/fm/fm-train .
ln -sf solvers/fm/fm-predict .
#make -C solvers/fm-sse-linear
#ln -sf solvers/fm-sse-linear/fm-sse-linear-train .
#ln -sf solvers/fm-sse-linear/fm-sse-linear-predict .
#make -C solvers/fm-sse-sp3
#ln -sf solvers/fm-sse-sp3/fm-sse-sp3-train .
#ln -sf solvers/fm-sse-sp3/fm-sse-sp3-predict .
#make -C solvers/fm-sse-poly2-v2
#ln -sf solvers/fm-sse-poly2-v2/fm-sse-poly2-v2-train .
#ln -sf solvers/fm-sse-poly2-v2/fm-sse-poly2-v2-predict .
#make -C solvers/fm-sse-sp6
#ln -sf solvers/fm-sse-sp6/fm-sse-sp6-train .
#ln -sf solvers/fm-sse-sp6/fm-sse-sp6-predict .
#make -C solvers/fm-type2
#ln -sf solvers/fm-type2/fm-type2-train .
#ln -sf solvers/fm-type2/fm-type2-predict .
#make -C solvers/fm-sse-analysis
#ln -sf solvers/fm-sse-analysis/fm-sse-analysis-train .
#ln -sf solvers/fm-sse-analysis/fm-sse-analysis-predict .
#make -C solvers/fm-ccd
#ln -sf solvers/fm-ccd/fm-ccd-train .
#ln -sf solvers/fm-ccd/fm-ccd-predict .
#make -C solvers/fm-ccd-sgd
#ln -sf solvers/fm-ccd-sgd/fm-ccd-sgd-train .
#ln -sf solvers/fm-ccd-sgd/fm-ccd-sgd-predict .
#make -C solvers/fm-sp11
#ln -sf solvers/fm-sp11/fm-sp11-train .
#ln -sf solvers/fm-sp11/fm-sp11-predict .
#make -C solvers/fm-newton
#ln -sf solvers/fm-newton/fm-newton-train .
#ln -sf solvers/fm-newton/fm-newton-predict .
#make -C solvers/fm-ccd-v2
#ln -sf solvers/fm-ccd-v2/fm-ccd-v2-train .
#ln -sf solvers/fm-ccd-v2/fm-ccd-v2-predict .
#make -C solvers/fm-ccd-v3
#ln -sf solvers/fm-ccd-v3/fm-ccd-v3-train .
#ln -sf solvers/fm-ccd-v3/fm-ccd-v3-predict .
#make -C solvers/fm-sp12
#ln -sf solvers/fm-sp12/fm-sp12-train .
#ln -sf solvers/fm-sp12/fm-sp12-predict .
#make -C solvers/fm-sse-tensor3/
#ln -sf solvers/fm-sse-tensor3/fm-sse-tensor3-train .
#ln -sf solvers/fm-sse-tensor3/fm-sse-tensor3-predict .
#make -C solvers/fm-tensor3-v2/
#ln -sf solvers/fm-tensor3-v2/fm-tensor3-v2-train .
#ln -sf solvers/fm-tensor3-v2/fm-tensor3-v2-predict .
#make -C solvers/fm-mark1/
#ln -sf solvers/fm-mark1/fm-mark1-train .
#ln -sf solvers/fm-mark1/fm-mark1-predict .
#make -C solvers/fm-mark2/
#ln -sf solvers/fm-mark2/fm-mark2-train .
#ln -sf solvers/fm-mark2/fm-mark2-predict .
make -C solvers/mark24/
ln -sf solvers/mark24/mark24-train .
ln -sf solvers/mark24/mark24-predict .

ln -sf trva.csv tr.r0.csv
ln -sf te.csv va.r0.csv

if [[ "$(hostname)" == linux* ]]; then
ln -sf /tmp2/r01922136/depo/data/criteo/* .
rm -f defender.txt
elif [[ "$(hostname)" == optima ]] || [[ "$(hostname)" == schroeder ]]; then
ln -sf /tmp2/criteo/* .
rm -f defender.txt
fi
