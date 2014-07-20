#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

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
    Model(size_t const n, size_t const k) : n(n), k(k), P(n*k), W(n, 0) {} 
    size_t const n, k; 
    std::vector<float> P, W;
};

void save_model(Model const &model, std::string const &path);

Model read_model(std::string const &path);

SpMat read_data(std::string const tr_path);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline float calc_rate(
    size_t const i, 
    Model const &model, 
    size_t const * const jv_begin, 
    size_t const * const jv_end)
{
    size_t const k = model.k;
    size_t const n = model.n;
    float const * const P = model.P.data();
    float const * const W = model.W.data();
    
    float r = 0;
    for(size_t const *u = jv_begin; u != jv_end; ++u)
    {
        if(*u >= n)
            continue;
        float const * const pu = P+(*u)*k;
        for(size_t const *v = u+1; v != jv_end; ++v) 
        {
            if(*v >= n)
                continue;
            float const * const pv = P+(*v)*k;
            for(size_t d = 0; d < k; ++d)
                r += (*(pu+d))*(*(pv+d));
        }

        float const * const wu = W+(*u);
        r += (*wu);
    }

    return r;
}

#endif // _COMMON_H_
