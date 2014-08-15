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
size_t const kF_SIZE = 39;

struct W_Node
{
    W_Node() : w(0), wg(1) {}
    float w, wg;
};

struct W_Vector
{
    W_Vector(size_t const k) : wv(kF_SIZE*k) {}
    std::vector<W_Node> wv;
    W_Node & operator [] (size_t idx) {return wv[idx];}
};

struct Model
{
    Model(size_t const k) : W(kW_SIZE, k), k(k) {}
    std::vector<W_Vector> W;
    size_t k;
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
    float const kappa_x1_x2, float const eta, float const lambda)
{
    float const g1 = lambda*w1.w + kappa_x1_x2*w2.w;
    float const g2 = lambda*w2.w + kappa_x1_x2*w1.w;

    w1.wg += g1*g1;
    w2.wg += g2*g2;

    w1.w -= eta*qrsqrt(w1.wg)*g1;
    w2.w -= eta*qrsqrt(w2.wg)*g2;
}

inline float wTx(SpMat const &problem, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
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

            for(size_t d = 0; d < model.k; ++d)
            {
                W_Node &w1 = model.W[j1%kW_SIZE][f2*model.k+d];
                W_Node &w2 = model.W[j2%kW_SIZE][f1*model.k+d];
                if(do_update)
                    update(w1, w2, kappa*x1*x2, eta, lambda);
                else
                    t += w1.w*w2.w*x1*x2;
            }
        }
    }
    return t;
}

float predict(SpMat const &problem, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
