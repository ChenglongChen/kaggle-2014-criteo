#!/bin/bash

make -C solvers/mark29/
ln -sf solvers/mark29/mark29 .
make -C solvers/mark33/
ln -sf solvers/mark33/mark33 .
