#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>

struct SpMat
{
    SpMat() : n(0) {}
    std::vector<int> Y;
    std::vector<size_t> P, J;
    std::vector<double> X;
    size_t n;
};

SpMat read_data(std::string const tr_path);

size_t const kW_SIZE = 1e+7;

struct Model
{
    Model() : W(kW_SIZE, 0), WG(kW_SIZE, 0) {}
    std::vector<double> W, WG;
};

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
