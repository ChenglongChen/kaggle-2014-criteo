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
    size_t n;
};

SpMat read_data(std::string const tr_path);

struct Model
{
    Model(size_t const n) : W(n, 0), WG(n, 0) {}
    std::vector<double> W, WG;
};

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
