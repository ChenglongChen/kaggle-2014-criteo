#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

#include <pmmintrin.h>

size_t const kNR_FEAT = 13;

struct Node
{
    Node(size_t const j, float const v) : j(j), v(v) {}
    size_t j;
    float v;
};

struct Mat
{
    Mat(size_t const nr_instance) 
        : nr_instance(0), 
          X(kNR_FEAT, std::vector<float>(nr_instance, 0)), 
          Y(nr_instance, 0) {}
    size_t  nr_instance;
    std::vector<std::vector<float>> X;
    std::vector<float> Y;
};

Mat read_data(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
