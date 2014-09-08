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

struct DenseColMat
{
    DenseColMat(size_t const nr_instance) 
        : nr_instance(nr_instance), 
          X(kNR_FEATURE, std::vector<float>(nr_instance)), 
          Y(nr_instance) {}
    size_t  nr_instance;
    std::vector<std::vector<float>> X;
    std::vector<float> Y;
};

DenseColMat read_data(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
