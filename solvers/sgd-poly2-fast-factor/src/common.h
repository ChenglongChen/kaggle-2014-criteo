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
    Node(size_t const f, size_t const j, float const x) : f(f), j(j), x(x) {}
    size_t f, j;
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

size_t const kW_SIZE = 1e+7;
size_t const kNR_FIELDS = 39;

struct W_Node
{
    W_Node() : w(0), wg(0) {}
    float w, wg;
};

struct W_Vector
{
    W_Vector() : wv(kNR_FIELDS) {}
    std::vector<W_Node> wv;
};

struct Model
{
    Model() : W(kW_SIZE) {}
    std::vector<W_Vector> W;
};

void save_model(Model const &model, std::string const &path);

Model load_model(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline size_t calc_w_idx(size_t const a, size_t const b)
{
    return ((a+b)*(a+b+1)/2+b)%kW_SIZE;
}

inline float qrsqrt(float x)
{
    _mm_store_ss(&x, _mm_rsqrt_ps(_mm_load1_ps(&x)));
    return x;
}

inline void update(W_Node &w1, W_Node &w2, 
    float const kappa_x1_x2, float const eta)
{
    float const g1 = kappa_x1_x2*w2.w;
    float const g2 = kappa_x1_x2*w1.w;

    w1.wg += g1;
    w2.wg += g2;

    w1.w -= eta*qrsqrt(w1.wg)*g1;
    w2.w -= eta*qrsqrt(w2.wg)*g2;
}

inline float wTx(SpMat const &problem, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, bool const do_update=false)
{
    float t = 0;
    for(size_t idx1 = problem.P[i]; idx1 < problem.P[i+1]; ++idx1)
    {
        size_t const j1 = problem.JX[idx1].j;
        size_t const f1 = problem.JX[idx1].f;
        float const x1 = problem.JX[idx1].x;
        for(size_t idx2 = idx1+1; idx2 < problem.P[i+1]; ++idx2)
        {
            size_t const j2 = problem.JX[idx2].j;
            size_t const f2 = problem.JX[idx2].f;
            float const x2 = problem.JX[idx2].x;

            W_Node &w1 = model.W[j1%kW_SIZE].wv[f2];
            W_Node &w2 = model.W[j2%kW_SIZE].wv[f1];

            if(do_update)
                update(w1, w2, kappa*x1*x2, eta);
            else
                t += w1.w*w2.w*x1*x2;
        }
    }
    return t;
}

float predict(SpMat const &problem, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
