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

size_t const kF_SIZE = 40;
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
    __m128 const XMMkappa = _mm_load1_ps(&kappa);
    __m128 const XMMeta = _mm_load1_ps(&eta);
    __m128 const XMMlambda = _mm_load1_ps(&lambda);

    __m128 XMMt = _mm_setzero_ps();
    for(size_t idx1 = problem.P[i]; idx1 < problem.P[i+1]; ++idx1)
    {
        size_t const j1 = problem.JX[idx1].j;
        size_t const f1 = problem.JX[idx1].f;
        __m128 const XMMx1 = _mm_load1_ps(&problem.JX[idx1].x);
        __m128 const XMMkappa_x1 = _mm_mul_ps(XMMkappa, XMMx1);

        for(size_t idx2 = idx1+1; idx2 < problem.P[i+1]; ++idx2)
        {
            size_t const j2 = problem.JX[idx2].j;
            size_t const f2 = problem.JX[idx2].f;
            __m128 const XMMx2 = _mm_load1_ps(&problem.JX[idx2].x);
            __m128 const XMMkappa_x1_x2 = _mm_mul_ps(XMMkappa_x1, XMMx2);

            float * const w1 = 
                model.W.data()+j1*kF_SIZE*k*kW_NODE_SIZE+f2*k*kW_NODE_SIZE;
            float * const w2 = 
                model.W.data()+j2*kF_SIZE*k*kW_NODE_SIZE+f1*k*kW_NODE_SIZE;

            if(do_update)
            {
                for(size_t d = 0; d < k; d += 4)
                {
                    __m128 XMMw1 = _mm_load_ps(w1+d);
                    __m128 XMMw2 = _mm_load_ps(w2+d);

                    __m128 XMMwg1 = _mm_load_ps(w1+k+d);
                    __m128 XMMwg2 = _mm_load_ps(w2+k+d);

                    __m128 XMMg1 = _mm_add_ps(
                        _mm_mul_ps(XMMlambda, XMMw1),
                        _mm_mul_ps(XMMkappa_x1_x2, XMMw2));
                    __m128 XMMg2 = _mm_add_ps(
                        _mm_mul_ps(XMMlambda, XMMw2),
                        _mm_mul_ps(XMMkappa_x1_x2, XMMw1));

                    XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
                    XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

                    XMMw1 = _mm_sub_ps(XMMw1,
                        _mm_mul_ps(XMMeta, 
                        _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
                    XMMw2 = _mm_sub_ps(XMMw2,
                        _mm_mul_ps(XMMeta, 
                        _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

                    _mm_store_ps(w1+d, XMMw1);
                    _mm_store_ps(w2+d, XMMw2);

                    _mm_store_ps(w1+k+d, XMMwg1);
                    _mm_store_ps(w2+k+d, XMMwg2);
                }
            }
            else
            {
                for(size_t d = 0; d < k; d += 4)
                {
                    __m128 const XMMw1 = _mm_load_ps(w1+d);
                    __m128 const XMMw2 = _mm_load_ps(w2+d);

                    XMMt = _mm_add_ps(XMMt, 
                        _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMx1), XMMx2));
                }
            }
        }
    }

    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    float t;
    _mm_store_ss(&t, XMMt);

    return t;
}

float predict(SpMat const &problem, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
