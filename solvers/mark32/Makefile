CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x -march=native -fopenmp
MAIN = mark32-train mark32-predict
FILES = common.cpp timer.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)

all: $(MAIN)

mark32-train: src/train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

mark32-predict: src/predict.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
