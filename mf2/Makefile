CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x 
MAIN = mf-train #mf-predict
FILES = common.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)

all: $(MAIN)

mf-train: src/train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

mf-predict: src/predict.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
