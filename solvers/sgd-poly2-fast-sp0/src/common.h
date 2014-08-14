#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

struct Node
{
    Node(size_t const j, float const x) : j(j), x(x) {}
    size_t j;
    float x;
};

struct SpMat
{
    SpMat() : n(0) {}
    std::vector<size_t> P;
    std::vector<Node> JX;
    std::vector<float> Y;
    size_t n;
};

SpMat read_data(std::string const tr_path);

size_t const kW_SIZE = 1e+8;

struct Model
{
    Model() : W(kW_SIZE*2, 0) {}
    std::vector<float> W;
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

inline float wTx(SpMat const &problem_i, SpMat const &problem_c, 
    Model const &model, size_t const i)
{
    float t = 0;
    for(size_t idx1 = problem_i.P[i]; idx1 < problem_i.P[i+1]; ++idx1)
    {
        size_t const j1 = problem_i.JX[idx1].j;
        float const x1 = problem_i.JX[idx1].x;

        for(size_t idx2 = idx1+1; idx2 < problem_i.P[i+1]; ++idx2)
        {
            size_t const j2 = problem_i.JX[idx2].j;
            float const x2 = problem_i.JX[idx2].x;
            size_t const w_idx = (cantor(j1,j2)%kW_SIZE)*2;
            t += model.W[w_idx]*x1*x2;
        }

        for(size_t idx2 = problem_c.P[i]; idx2 < problem_c.P[i+1]; ++idx2)
        {
            size_t const j2 = problem_c.JX[idx2].j;
            float const x2 = problem_c.JX[idx2].x;
            size_t const w_cdx = (cantor(j1,j2)%kW_SIZE)*2;
            t += model.W[w_cdx]*x1*x2;
        }
    }

    for(size_t idx1 = problem_c.P[i]; idx1 < problem_c.P[i+1]; ++idx1)
    {
        size_t const j1 = problem_c.JX[idx1].j;
        float const x1 = problem_c.JX[idx1].x;
        for(size_t idx2 = idx1+1; idx2 < problem_c.P[i+1]; ++idx2)
        {
            size_t const j2 = problem_c.JX[idx2].j;
            float const x2 = problem_c.JX[idx2].x;
            size_t const w_cdx = (cantor(j1,j2)%kW_SIZE)*2;
            t += model.W[w_cdx]*x1*x2;
        }
    }

    return t;
}

float predict(SpMat const &problem_i, SpMat const &problem_c, 
    Model const &model, std::string const &output_path = std::string(""));
#endif // _COMMON_H_