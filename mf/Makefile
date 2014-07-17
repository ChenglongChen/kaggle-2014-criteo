CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x 
MAIN = online-train online-predict
FILES = ftrl.cpp learner.cpp common.cpp eval.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)
#OBJS = $(SRCS:.c=.o)

all: $(MAIN)

online-train: src/online-train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

online-predict: src/online-predict.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
