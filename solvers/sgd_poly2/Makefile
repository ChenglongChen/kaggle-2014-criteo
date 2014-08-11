CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x 
MAIN = sgd-poly2-train #mf-predict
FILES = common.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)

all: $(MAIN)

sgd-poly2-train: src/train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
