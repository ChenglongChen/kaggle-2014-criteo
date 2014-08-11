#!/bin/bash

cd ~/work2/kaggle.2014.criteo/
make -C solvers/superliblinear/
ln -sf solvers/superliblinear/train supertrain
ln -sf solvers/superliblinear/predict superpredict
make -C solvers/sgd-poly2/
ln -sf solvers/sgd-poly2/sgd-poly2-train .
ln -sf solvers/sgd-poly2/sgd-poly2-predict .

if [[ "$(hostname)" == linux* ]]; then
ln -sf /tmp2/r01922136/depo/data/criteo/* .
rm -f defender.txt
elif [[ "$(hostname)" == optima ]]; then
ln -sf /tmp2/criteo/* .
rm -f defender.txt
fi
