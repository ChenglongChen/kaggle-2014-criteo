CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x 
MAIN = sgd-poly2-train #sgd-poly2-predict
FILES = common.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)

all: $(MAIN)

sgd-poly2-train: src/train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

#sgd-poly2-predict: src/predict.cpp $(SRCS) $(HEADERS)
#	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
