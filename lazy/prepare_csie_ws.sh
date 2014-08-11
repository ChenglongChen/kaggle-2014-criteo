#!/bin/bash

cd ~/work2/kaggle.2014.criteo/
./lazy/prepare.sh
ln -sf /tmp2/r01922136/depo/data/criteo/* .
rm -f defender.txt
