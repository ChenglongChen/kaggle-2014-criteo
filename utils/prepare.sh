#!/bin/bash

if [[ "$(hostname)" == linux* ]]; then
cd ~/work2/kaggle.2014.criteo/
elif [[ "$(hostname)" == optima ]] || [[ "$(hostname)" == schroeder ]]; then
cd ~/work/kaggle.2014.criteo/
fi

if [[ "$(hostname)" == linux* ]]; then
ln -sf /tmp2/r01922136/depo/data/criteo/* .
rm -f defender.txt
elif [[ "$(hostname)" == optima ]] || [[ "$(hostname)" == schroeder ]]; then
ln -sf /tmp2/criteo/* .
rm -f defender.txt
fi

ln -sf trva.csv tr.r0.csv
ln -sf te.csv va.r0.csv
