#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

#include <pmmintrin.h>

struct Problem
{
    Problem(uint32_t const nr_instance, uint32_t const nr_field) 
        : nr_instance(nr_instance), nr_field(nr_field), nr_sparse_field(0),
          X(nr_field, std::vector<float>(nr_instance)), 
          Y(nr_instance) {}
    uint32_t const nr_instance, nr_field;
    uint32_t nr_sparse_field;
    std::vector<std::vector<float>> X;
    std::vector<uint32_t> SX;
    std::vector<uint64_t> SP;
    std::vector<float> Y;
};

Problem read_data(std::string const &dense_path, std::string const &sparse_path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
