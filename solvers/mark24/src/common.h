#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

#include <pmmintrin.h>

struct Node
{
    Node(size_t const f, size_t const j, float const v) : f(f), j(j), v(v) {}
    size_t f, j;
    float v;
};

struct SpMat
{
    SpMat() : nr_feature(0), nr_instance(0) {}
    std::vector<size_t> P;
    std::vector<Node> X;
    std::vector<float> Y;
    size_t nr_feature, nr_instance;
};

SpMat read_data(std::string const path, size_t const reserved_size=0);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
