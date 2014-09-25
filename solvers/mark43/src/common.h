#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

#include <pmmintrin.h>

struct DenseColMat
{
    DenseColMat(uint64_t const nr_instance, uint64_t const nr_field) 
        : nr_instance(nr_instance), nr_field(nr_field),
          X(nr_field, std::vector<float>(nr_instance)), 
          Y(nr_instance) {}
    uint64_t const nr_instance, nr_field;
    std::vector<std::vector<float>> X;
    std::vector<float> Y;
};

DenseColMat read_dcm(std::string const &path);

struct SparseColMat
{
    SparseColMat(uint64_t const nr_instance, uint64_t const nr_field, uint64_t const nnz) 
        : nr_instance(nr_instance), nr_field(nr_field), nnz(nnz), 
          X(nnz), P(nr_field+1) {}
    uint64_t const nr_instance, nr_field, nnz;
    std::vector<uint64_t> X, P;
};

SparseColMat read_scm(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
