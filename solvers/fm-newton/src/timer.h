#include <chrono>

class Timer
{
public:
    Timer();
    void reset();
    void tic();
    double toc();
    double get();
private:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::milliseconds duration;
};
