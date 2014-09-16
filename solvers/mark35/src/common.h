#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

#include <pmmintrin.h>

struct SpMat
{
    SpMat(uint32_t const nr_instance, uint32_t const nr_field) 
        : nr_feature(0), nr_instance(nr_instance), nr_field(nr_field), 
          v(2.0f/static_cast<float>(nr_field)), J(nr_instance*nr_field), 
          Y(nr_instance) {}
    uint32_t nr_feature, nr_instance, nr_field;
    float v;
    std::vector<uint32_t> J;
    std::vector<float> Y;
};

SpMat read_data(std::string const path);

uint32_t const kW_NODE_SIZE = 2;
uint32_t const kNR_BIN = 1e+6;

struct WNode
{
    WNode() : v(0), sg2(1), mask(0) {}
    float v, sg2;
    uint32_t mask;
};

struct Model
{
    Model(uint32_t const nr_feature, uint32_t const nr_factor, uint32_t const nr_field) 
        : W(static_cast<uint64_t>(nr_feature)*nr_field*nr_factor*kW_NODE_SIZE, 0), 
          WP2(nr_feature), nr_feature(nr_feature), 
          nr_factor(nr_factor), nr_field(nr_field) {}
    std::vector<float> W;
    std::vector<WNode> WP2;
    const uint32_t nr_feature, nr_factor, nr_field;
};

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline float qrsqrt(float x)
{
    _mm_store_ss(&x, _mm_rsqrt_ps(_mm_load1_ps(&x)));
    return x;
}

inline uint32_t calc_w_idx(uint32_t const a, uint32_t const b)
{
    return ((a+b)*(a+b+1)/2+b)%kNR_BIN;
}

inline float wTx(SpMat const &spmat, Model &model, uint32_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    uint32_t const nr_factor = model.nr_factor;
    uint32_t const nr_field = model.nr_field;
    uint32_t const nr_feature = model.nr_feature;
    uint64_t const align0 = nr_factor*kW_NODE_SIZE;
    uint64_t const align1 = nr_field*align0;

    __m128 const XMMv = _mm_set1_ps(spmat.v);
    __m128 const XMMkappav = _mm_set1_ps(kappa*spmat.v);
    __m128 const XMMeta = _mm_load1_ps(&eta);
    __m128 const XMMlambda = _mm_load1_ps(&lambda);

    __m128 XMMt = _mm_setzero_ps();
    float tp2 = 0;
    for(uint32_t f1 = 0; f1 < nr_field; ++f1)
    {
        uint32_t const j1 = spmat.J[i*spmat.nr_field+f1];
        if(j1 >= nr_feature)
            continue;

        for(uint32_t f2 = f1+1; f2 < nr_field; ++f2)
        {
            uint32_t const j2 = spmat.J[i*spmat.nr_field+f2];
            if(j2 >= nr_feature)
                continue;

            float * const w1 = model.W.data() + j1*align1 + f2*align0;
            float * const w2 = model.W.data() + j2*align1 + f1*align0;

            if(do_update)
            {
                float * const wg1 = w1 + nr_factor;
                float * const wg2 = w2 + nr_factor;
                for(uint32_t d = 0; d < nr_factor; d += 4)
                {
                    __m128 XMMw1 = _mm_load_ps(w1+d);
                    __m128 XMMw2 = _mm_load_ps(w2+d);

                    __m128 XMMwg1 = _mm_load_ps(wg1+d);
                    __m128 XMMwg2 = _mm_load_ps(wg2+d);

                    __m128 XMMg1 = _mm_add_ps(
                        _mm_mul_ps(XMMlambda, XMMw1),
                        _mm_mul_ps(XMMkappav, XMMw2));
                    __m128 XMMg2 = _mm_add_ps(
                        _mm_mul_ps(XMMlambda, XMMw2),
                        _mm_mul_ps(XMMkappav, XMMw1));

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

                    _mm_store_ps(wg1+d, XMMwg1);
                    _mm_store_ps(wg2+d, XMMwg2);
                }
            }
            else
            {
                for(uint32_t d = 0; d < nr_factor; d += 4)
                {
                    __m128 const XMMw1 = _mm_load_ps(w1+d);
                    __m128 const XMMw2 = _mm_load_ps(w2+d);

                    XMMt = _mm_add_ps(XMMt, _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
                }
            }

            if(f1 > 38 or f2 > 38)
                continue;

            WNode &w = model.WP2[calc_w_idx(j1, j2)];
            if(w.mask)
            {
                if(do_update)
                {
                    float const g = lambda*w.v + kappa*spmat.v;

                    w.sg2 += g*g;

                    w.v -= eta*qrsqrt(w.sg2)*g;
                }
                else
                {
                    tp2 += w.v*spmat.v;
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

/*
inline float wTx(SpMat const &spmat, Model &model, uint32_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    uint32_t const nr_factor = model.nr_factor;
    uint32_t const nr_field = model.nr_field;

    float t = 0;
    for(uint32_t idx1 = spmat.P[i]; idx1 < spmat.P[i+1]; ++idx1)
    {
        uint32_t const j1 = spmat.X[idx1].j;
        uint32_t const f1 = spmat.X[idx1].f;
        float const v1 = spmat.X[idx1].v;

        for(uint32_t idx2 = idx1+1; idx2 < spmat.P[i+1]; ++idx2)
        {
            uint32_t const j2 = spmat.X[idx2].j;
            uint32_t const f2 = spmat.X[idx2].f;
            float const v2 = spmat.X[idx2].v;

            float * w1 = 
                model.W.data()+j1*nr_field*nr_factor*kW_NODE_SIZE+f2*nr_factor*kW_NODE_SIZE;
            float * w2 = 
                model.W.data()+j2*nr_field*nr_factor*kW_NODE_SIZE+f1*nr_factor*kW_NODE_SIZE;

            if(do_update)
            {
                float * wg1 = w1 + nr_factor; 
                float * wg2 = w2 + nr_factor; 
                for(uint32_t d = 0; d < nr_factor; ++d, ++w1, ++w2, ++wg1, ++wg2)
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
                for(uint32_t d = 0; d < nr_factor; ++d, ++w1, ++w2)
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
