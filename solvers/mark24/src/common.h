#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

#include <pmmintrin.h>

size_t const kNR_FEATURE = 13;

struct Node
{
    Node() : i(0), v(0) {}
    Node(size_t const i, float const v) : i(i), v(v) {}
    size_t i;
    float v;
};

struct Problem
{
    Problem(size_t const nr_instance) 
        : nr_instance(0), 
          X(kNR_FEATURE, std::vector<Node>(nr_instance)), 
          Y(nr_instance, 0) {}
    size_t  nr_instance;
    std::vector<std::vector<Node>> X;
    std::vector<float> Y;
};

Problem read_data(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
