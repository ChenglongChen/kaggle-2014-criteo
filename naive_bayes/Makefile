CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x 
MAIN = train extract predict
FILES = ftrl.cpp learner.cpp common.cpp eval.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)
#OBJS = $(SRCS:.c=.o)

all: $(MAIN)

train: src/train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

extract: src/extract.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

predict: src/predict.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
