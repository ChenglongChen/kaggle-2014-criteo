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
make -C solvers/fm
ln -sf solvers/fm/fm-train .
ln -sf solvers/fm/fm-predict .
make -C solvers/fm-sse
ln -sf solvers/fm-sse/fm-sse-train .
ln -sf solvers/fm-sse/fm-sse-predict .
make -C solvers/fm-sse-linear
ln -sf solvers/fm-sse-linear/fm-sse-linear-train .
ln -sf solvers/fm-sse-linear/fm-sse-linear-predict .

ln -sf trva.csv tr.r0.csv
ln -sf te.csv va.r0.csv

if [[ "$(hostname)" == linux* ]]; then
ln -sf /tmp2/r01922136/depo/data/criteo/* .
rm -f defender.txt
elif [[ "$(hostname)" == optima ]] || [[ "$(hostname)" == schroeder ]]; then
ln -sf /tmp2/criteo/* .
rm -f defender.txt
fi
