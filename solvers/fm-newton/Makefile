CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x -march=native -fopenmp
MAIN = fm-newton-train fm-newton-predict
FILES = common.cpp timer.cpp tron.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)
LIBS = blas/blas.a

all: $(MAIN)

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

fm-newton-train: src/train.cpp $(SRCS) $(HEADERS) $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS) $(LIBS)

fm-newton-predict: src/predict.cpp $(SRCS) $(HEADERS) $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS) $(LIBS)

clean:
	rm -f $(MAIN)
