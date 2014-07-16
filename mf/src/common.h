#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _UTIL_H_
#define _UTIL_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>

typedef unsigned int uint;
typedef unsigned long long ull;

FILE *open_c_file(std::string const &path, std::string const &mode);

struct SpMat
{
    SpMat() : n(0) {}
    std::vector<int> yv;
    std::vector<size_t> pv, jv;
    size_t n;
};

struct Model
{
    Model(size_t const n, size_t const k) : n(n), k(k), P(n*k) {} 
    size_t const n, k; 
    std::vector<float> P;
};

void save_model(Model const &model, std::string const &path);

Model read_model(std::string const &path);

SpMat read_data(std::string const tr_path);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _UTIL_H_