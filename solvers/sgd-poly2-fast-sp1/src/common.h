#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

#include <pmmintrin.h>

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

struct WNode
{
    WNode() : w(0), wg(0) {}
    float w, wg;
};

struct Model
{
    Model() : W(kW_SIZE) {}
    std::vector<WNode> W;
};

void save_model(Model const &model, std::string const &path);

Model load_model(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline size_t calc_w_idx(size_t const a)
{
    return a%kW_SIZE;
}

inline size_t calc_w_idx(size_t const a, size_t const b)
{
    return ((a+b)*(a+b+1)/2+b)%kW_SIZE;
}

inline float qrsqrt(float x)
{
    _mm_store_ss(&x, _mm_rsqrt_ps(_mm_load1_ps(&x)));
    return x;
}

inline void update(Model &model, size_t const w_idx, float const g, float const eta)
{
    WNode &w1 = model.W[w_idx];
    w1.wg += g*g;
    w1.w -= eta*qrsqrt(w1.wg)*g;
}

inline float wTx_p1(SpMat const &problem, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, bool const do_update=false)
{
    float t = 0;
    for(size_t idx1 = problem.P[i]; idx1 < problem.P[i+1]; ++idx1)
    {
        size_t const j1 = problem.JX[idx1].j;
        float const x1 = problem.JX[idx1].x;

        size_t const w_idx = calc_w_idx(j1);

        if(do_update)
            update(model, w_idx, kappa*x1, eta);
        else
            t += model.W[w_idx].w*x1;
    }
    return t;
}

inline float wTx_p2(SpMat const &problem, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, bool const do_update=false)
{
    float t = 0;
    for(size_t idx1 = problem.P[i]; idx1 < problem.P[i+1]; ++idx1)
    {
        size_t const j1 = problem.JX[idx1].j;
        float const x1 = problem.JX[idx1].x;
        for(size_t idx2 = idx1+1; idx2 < problem.P[i+1]; ++idx2)
        {
            size_t const j2 = problem.JX[idx2].j;
            float const x2 = problem.JX[idx2].x;

            size_t const w_idx = calc_w_idx(j1,j2);

            if(do_update)
                update(model, w_idx, kappa*x1*x2, eta);
            else
                t += model.W[w_idx].w*x1*x2;
        }
    }
    return t;
}

float predict(SpMat const &problem, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
