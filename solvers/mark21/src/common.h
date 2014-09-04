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
    Node(size_t const f, size_t const j, float const v) : f(f), j(j), v(v) {}
    size_t f, j;
    float v;
};

struct SpMat
{
    SpMat() : nr_feature(0), nr_instance(0) {}
    std::vector<size_t> P;
    std::vector<Node> X;
    std::vector<float> Y;
    size_t nr_feature, nr_instance;
};

SpMat read_data(std::string const path);

size_t const kNR_FIELD = 39;
size_t const kW_NODE_SIZE = 2;

struct Model
{
    Model(size_t const nr_feature, size_t const nr_factor, size_t const nr_factor2) 
        : W(nr_feature*kNR_FIELD*nr_factor*kW_NODE_SIZE, 0), 
          WA(nr_feature*nr_factor2*kW_NODE_SIZE, 0),
          nr_feature(nr_feature), nr_factor(nr_factor), 
          nr_factor2(nr_factor2) {}
    std::vector<float> W, WA;
    const size_t nr_feature, nr_factor, nr_factor2;
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

inline float wTx(SpMat const &spmat, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    size_t const nr_factor = model.nr_factor;
    __m128 const XMMkappa = _mm_load1_ps(&kappa);
    __m128 const XMMeta = _mm_load1_ps(&eta);
    __m128 const XMMlambda = _mm_load1_ps(&lambda);

    __m128 XMMt = _mm_setzero_ps();
    for(size_t idx1 = spmat.P[i]; idx1 < spmat.P[i+1]; ++idx1)
    {
        size_t const j1 = spmat.X[idx1].j;
        if(j1 >= model.nr_feature)
            continue;
        size_t const f1 = spmat.X[idx1].f;
        __m128 const XMMv1 = _mm_load1_ps(&spmat.X[idx1].v);
        __m128 const XMMkappa_v1 = _mm_mul_ps(XMMkappa, XMMv1);

        for(size_t idx2 = idx1+1; idx2 < spmat.P[i+1]; ++idx2)
        {
            size_t const j2 = spmat.X[idx2].j;
            if(j2 >= model.nr_feature)
                continue;
            size_t const f2 = spmat.X[idx2].f;
            __m128 const XMMv2 = _mm_load1_ps(&spmat.X[idx2].v);
            __m128 const XMMkappa_v1_v2 = _mm_mul_ps(XMMkappa_v1, XMMv2);

            float * const w1 = 
                model.W.data()+j1*kNR_FIELD*nr_factor*kW_NODE_SIZE+f2*nr_factor*kW_NODE_SIZE;
            float * const w2 = 
                model.W.data()+j2*kNR_FIELD*nr_factor*kW_NODE_SIZE+f1*nr_factor*kW_NODE_SIZE;

            if(do_update)
            {
                for(size_t d = 0; d < nr_factor; d += 4)
                {
                    __m128 XMMw1 = _mm_load_ps(w1+d);
                    __m128 XMMw2 = _mm_load_ps(w2+d);

                    __m128 XMMwg1 = _mm_load_ps(w1+nr_factor+d);
                    __m128 XMMwg2 = _mm_load_ps(w2+nr_factor+d);

                    __m128 XMMg1 = _mm_add_ps(
                        _mm_mul_ps(XMMlambda, XMMw1),
                        _mm_mul_ps(XMMkappa_v1_v2, XMMw2));
                    __m128 XMMg2 = _mm_add_ps(
                        _mm_mul_ps(XMMlambda, XMMw2),
                        _mm_mul_ps(XMMkappa_v1_v2, XMMw1));

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

                    _mm_store_ps(w1+nr_factor+d, XMMwg1);
                    _mm_store_ps(w2+nr_factor+d, XMMwg2);
                }
            }
            else
            {
                for(size_t d = 0; d < nr_factor; d += 4)
                {
                    __m128 const XMMw1 = _mm_load_ps(w1+d);
                    __m128 const XMMw2 = _mm_load_ps(w2+d);

                    XMMt = _mm_add_ps(XMMt, 
                        _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv1), XMMv2));
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

inline float wTx2(SpMat const &problem, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    size_t const k = model.nr_factor2;
    __m128 const XMMkappa = _mm_load1_ps(&kappa);
    __m128 const XMMeta = _mm_load1_ps(&eta);
    __m128 const XMMlambda = _mm_load1_ps(&lambda);

    std::vector<float> sv(k, 0);
    float * const s = sv.data();

    __m128 XMMt = _mm_setzero_ps();
    for(size_t idx = problem.P[i]; idx < problem.P[i+1]; ++idx)
    {
        size_t const j = problem.X[idx].j;
        __m128 const XMMx = _mm_load1_ps(&problem.X[idx].v);
        float * const w = model.WA.data()+j*k*kW_NODE_SIZE;

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
        size_t const j = problem.X[idx].j;
        __m128 const XMMx = _mm_load1_ps(&problem.X[idx].v);
        float * const w = model.WA.data()+j*k*kW_NODE_SIZE;
        
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

/*
inline float wTx(SpMat const &spmat, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    size_t const nr_factor = model.nr_factor;

    float t = 0;
    for(size_t idx1 = spmat.P[i]; idx1 < spmat.P[i+1]; ++idx1)
    {
        size_t const j1 = spmat.X[idx1].j;
        size_t const f1 = spmat.X[idx1].f;
        float const v1 = spmat.X[idx1].v;

        for(size_t idx2 = idx1+1; idx2 < spmat.P[i+1]; ++idx2)
        {
            size_t const j2 = spmat.X[idx2].j;
            size_t const f2 = spmat.X[idx2].f;
            float const v2 = spmat.X[idx2].v;

            float * w1 = 
                model.W.data()+j1*kNR_FIELD*nr_factor*kW_NODE_SIZE+f2*nr_factor*kW_NODE_SIZE;
            float * w2 = 
                model.W.data()+j2*kNR_FIELD*nr_factor*kW_NODE_SIZE+f1*nr_factor*kW_NODE_SIZE;

            if(do_update)
            {
                float * wg1 = w1 + nr_factor; 
                float * wg2 = w2 + nr_factor; 
                for(size_t d = 0; d < nr_factor; ++d, ++w1, ++w2, ++wg1, ++wg2)
                {
                    float const g1 = lambda*(*w1) + kappa*v1*v2*(*w2);
                    float const g2 = lambda*(*w2) + kappa*v1*v2*(*w1);

                    *wg1 += g1*g1;
                    *wg2 += g2*g2;

                    *w1 -= eta*qrsqrt(*wg1)*g1;
                    *w2 -= eta*qrsqrt(*wg2)*g2;
                }
            }
            else
            {
                for(size_t d = 0; d < nr_factor; ++d, ++w1, ++w2)
                    t += (*w1)*(*w2)*v1*v2;
            }
        }
    }

    return t;
}
*/

float predict(SpMat const &spmat, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
