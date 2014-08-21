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

size_t const kW_NODE_SIZE = 2;

struct Model
{
    Model(size_t const n, size_t const k) : W(n*k*kW_NODE_SIZE, 0), n(n), k(k) {}
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

    std::vector<float> sv(k, 0);
    float * const s = sv.data();

    __m128 XMMt = _mm_setzero_ps();
    for(size_t idx = problem.P[i]; idx < problem.P[i+1]; ++idx)
    {
        size_t const j = problem.JX[idx].j;
        __m128 const XMMx = _mm_load1_ps(&problem.JX[idx].x);
        float * const w = model.W.data()+j*k*kW_NODE_SIZE;

        for(size_t d = 0; d < k; d += 4)
        {
            __m128 XMMs = _mm_load_ps(s+d);
            __m128 const XMMw = _mm_load_ps(w+d);

            XMMs = _mm_add_ps(XMMs, _mm_mul_ps(XMMw, XMMx));

            _mm_store_ps(s+d, XMMs);
        }
    }

    for(size_t idx = problem.P[i]; idx < problem.P[i+1]; ++idx)
    {
        size_t const j = problem.JX[idx].j;
        __m128 const XMMx = _mm_load1_ps(&problem.JX[idx].x);
        float * const w = model.W.data()+j*k*kW_NODE_SIZE;
        
        if(do_update)
        {
            for(size_t d = 0; d < k; d += 4)
            {
                __m128 XMMs = _mm_load_ps(s+d);
                __m128 XMMw = _mm_load_ps(w+d);
                __m128 XMMwg = _mm_load_ps(w+k+d);

                __m128 XMMg = _mm_add_ps(_mm_mul_ps(XMMlambda, XMMw), 
                    _mm_mul_ps(XMMkappa, _mm_mul_ps(_mm_sub_ps(XMMs, _mm_mul_ps(XMMw, XMMx)), XMMx)));

                XMMwg = _mm_add_ps(XMMwg, _mm_mul_ps(XMMg, XMMg));

                XMMw = _mm_sub_ps(XMMw,
                    _mm_mul_ps(XMMeta, 
                    _mm_mul_ps(_mm_rsqrt_ps(XMMwg), XMMg)));

                _mm_store_ps(w+d, XMMw);
                _mm_store_ps(w+k+d, XMMwg);
            }
        }
        else
        {
            for(size_t d = 0; d < k; d += 4)
            {
                __m128 XMMs = _mm_load_ps(s+d);
                __m128 XMMw = _mm_load_ps(w+d);
                __m128 XMMwx = _mm_mul_ps(XMMw, XMMx);
                XMMt = _mm_add_ps(XMMt, 
                    _mm_mul_ps(XMMwx, _mm_sub_ps(XMMs, XMMwx)));
            }
        }
    }

    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    float t;
    _mm_store_ss(&t, XMMt);

    return t/2.0f;
}

float predict(SpMat const &problem, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
