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

SpMat read_data(std::string const path);

size_t const kF_SIZE = 39;
size_t const kW_NODE_SIZE = 2;

struct Model
{
    Model(size_t const n, size_t const k) : W(n*kF_SIZE*k*kW_NODE_SIZE, 0), n(n), k(k) {}
    std::vector<float> W;
    const size_t n, k;
};

void save_model(Model const &model, std::string const &path);

Model load_model(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline float qrsqrt(float x)
{
    _mm_store_ss(&x, _mm_rsqrt_ps(_mm_load1_ps(&x)));
    return x;
}

inline float wTx(SpMat const &problem, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    size_t const k = model.k;

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

            float * w1 = 
                model.W.data()+j1*kF_SIZE*k*kW_NODE_SIZE+f2*k*kW_NODE_SIZE;
            float * w2 = 
                model.W.data()+j2*kF_SIZE*k*kW_NODE_SIZE+f1*k*kW_NODE_SIZE;

            if(do_update)
            {
                float * wg1 = w1 + k;
                float * wg2 = w2 + k;
                for(size_t d = 0; d < k; ++d, ++w1, ++w2, ++wg1, ++wg2)
                {
                    float const g1 = lambda*(*w1) + kappa*x1*x2*(*w2);
                    float const g2 = lambda*(*w2) + kappa*x1*x2*(*w1);

                    *wg1 += g1*g1;
                    *wg2 += g2*g2;

                    *w1 -= eta*qrsqrt(*wg1)*g1;
                    *w2 -= eta*qrsqrt(*wg2)*g2;
                }
            }
            else
            {
                for(size_t d = 0; d < k; ++d, ++w1, ++w2)
                    t += (*w1)*(*w2)*x1*x2;
            }
        }
    }

    return t;
}

float predict(SpMat const &problem, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
