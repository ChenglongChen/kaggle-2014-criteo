CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x
LIBS ?= blas/blas.a
#LIBS ?= -lblas

all: supertrain-fm superpredict-fm

supertrain-fm: tron.o linear.o train.c blas/blas.a eval.o
	$(CXX) $(CFLAGS) -o supertrain-fm train.c tron.o eval.o linear.o $(LIBS)

superpredict-fm: tron.o linear.o predict.c blas/blas.a eval.o
	$(CXX) $(CFLAGS) -o superpredict-fm predict.c tron.o eval.o linear.o $(LIBS)

eval.o: eval.h eval.cpp
	$(CXX) $(CFLAGS) -c -o eval.o eval.cpp

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a:
	cd blas; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	cd blas;	make clean
	rm -f *~ tron.o linear.o supertrain-fm superpredict-fm
