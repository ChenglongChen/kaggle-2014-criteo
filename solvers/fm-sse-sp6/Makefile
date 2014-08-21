CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x -march=native -fopenmp
MAIN = fm-sse-sp6-train fm-sse-sp6-predict
FILES = common.cpp timer.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)

all: $(MAIN)

fm-sse-sp6-train: src/train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

fm-sse-sp6-predict: src/predict.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
