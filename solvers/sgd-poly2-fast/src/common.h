#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

struct SpMat
{
    SpMat() : n(0) {}
    std::vector<int> Y;
    std::vector<size_t> P, J;
    std::vector<float> X;
    size_t n;
};

SpMat read_data(std::string const tr_path);

size_t const kW_SIZE = 1e+7;

struct Model
{
    Model() : W(kW_SIZE, 0), WG(kW_SIZE, 0) {}
    std::vector<float> W, WG;
};

void save_model(Model const &model, std::string const &path);

Model load_model(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline size_t cantor(size_t const a, size_t const b)
{
    return (a+b)*(a+b+1)/2+b;
}

inline float logistic_func(float const t)
{
    return 1/(1+static_cast<float>(exp(-t)));
}

inline float wTx(SpMat const &problem, Model const &model, size_t const i)
{
    float t = 0;
    for(size_t idx1 = problem.P[i]; idx1 < problem.P[i+1]; ++idx1)
    {
        size_t const j1 = problem.J[idx1];
        float const x1 = problem.X[idx1];
        for(size_t idx2 = idx1+1; idx2 < problem.P[i+1]; ++idx2)
        {
            size_t const w_idx = cantor(j1,problem.J[idx2])%kW_SIZE;
            t += model.W[w_idx]*x1*problem.X[idx2];
        }
    }
    return t;
}

#endif // _COMMON_H_
