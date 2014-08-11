#!/bin/bash

cd ~/work2/kaggle.2014.criteo/
make -C solvers/superliblinear/
ln -s solvers/superliblinear/train supertrain
ln -s solvers/superliblinear/predict superpredict
make -C solvers/sgd_poly2/
ln -s solvers/sgd_poly2/sgd-poly2-train .
