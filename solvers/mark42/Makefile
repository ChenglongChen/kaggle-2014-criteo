CXX = g++
CXXFLAGS = -Wall -Wconversion -O2 -fPIC -std=c++0x -march=native -fopenmp
MAIN = mark42
FILES = common.cpp timer.cpp gbdt.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)

all: $(MAIN)

mark42: src/train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
