CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x -march=native
MAIN = sgd-poly3-train-fast sgd-poly3-predict-fast
FILES = common.cpp timer.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)

all: $(MAIN)

sgd-poly3-train-fast: src/train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

sgd-poly3-predict-fast: src/predict.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
