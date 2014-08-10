#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>

extern std::vector<size_t> const FieldSizes;
extern std::vector<size_t> const A;
extern std::vector<size_t> const B;

extern float const ALPHA;
extern float const BETA;
extern float const GAMMA;

struct SpMat
{
    std::vector<int> Y;
    std::vector<size_t> X;
};

SpMat read_data(std::string const tr_path);

struct Model
{
    Model(size_t const k, std::vector<size_t> const &fields_sizes) 
        : k(k)
    {
        for(auto size : fields_sizes)
        {
            W.emplace_back(size, 0); 
            WG.emplace_back(size, 0); 
        }
        for(size_t u = 0; u < fields_sizes.size(); ++u)
        {
            for(size_t v = u+1; v < fields_sizes.size(); ++v)
            {
                P.emplace_back(fields_sizes[u]*k);
                PG.emplace_back(fields_sizes[u]*k, 0);
                Q.emplace_back(fields_sizes[v]*k);
                QG.emplace_back(fields_sizes[v]*k, 0);
            }
        }
    }
    size_t const k; 
    std::vector<std::vector<float>> W, WG, P, PG, Q, QG;
};

void save_model(Model const &model, std::string const &path);

Model read_model(std::string const &path);

inline float calc_rate(
    Model const &model,
    size_t const * const x)
{
    float r = 0;
    for(auto f : A)
        r += model.W[f][x[f]]; 
    size_t cell = 0;
    for(size_t u = 0; u < B.size(); ++u)
        for(size_t v = u+1; v < B.size(); ++v, ++cell)
            for(size_t d = 0; d < model.k; ++d)
                r += model.P[cell][x[B[u]]*model.k+d]*model.Q[cell][x[B[v]]*model.k+d];
    return r;
}

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_