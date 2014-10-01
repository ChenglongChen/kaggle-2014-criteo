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
          v(2.0f/static_cast<float>(nr_field)), 
          J(static_cast<uint64_t>(nr_instance)*nr_field), 
          Y(nr_instance) {}
    uint32_t nr_feature, nr_instance, nr_field;
    float v;
    std::vector<uint32_t> J;
    std::vector<float> Y;
};

SpMat read_data(std::string const path);

uint32_t const kW_NODE_SIZE = 2;

struct Model
{
    Model(uint32_t const nr_feature, uint32_t const nr_factor, uint32_t const nr_field) 
        : W(static_cast<uint64_t>(nr_feature)*nr_field*nr_factor*kW_NODE_SIZE, 0), 
          nr_feature(nr_feature), nr_factor(nr_factor), nr_field(nr_field) {}
    std::vector<float> W;
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

/*
inline float wTx(SpMat const &spmat, Model &model, uint32_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    uint32_t const nr_factor = model.nr_factor;
    uint32_t const nr_field = model.nr_field;
    uint32_t const nr_feature = model.nr_feature;
    uint64_t const align0 = nr_factor*kW_NODE_SIZE;
    uint64_t const align1 = nr_field*align0;

    uint32_t const * const J = &spmat.J[i*nr_field];
    float * const W = model.W.data();

    __m128 const XMMv = _mm_set1_ps(spmat.v);
    __m128 const XMMkappav = _mm_set1_ps(kappa*spmat.v);
    __m128 const XMMeta = _mm_load1_ps(&eta);
    __m128 const XMMlambda = _mm_load1_ps(&lambda);

    __m128 XMMt = _mm_setzero_ps();
    for(uint32_t f1 = 0; f1 < 39; ++f1)
    {
        uint32_t const j1 = J[f1];
        if(j1 >= nr_feature)
            continue;

        for(uint32_t f2 = f1+1; f2 < 39; ++f2)
        {
            uint32_t const j2 = J[f2];
            if(j2 >= nr_feature)
                continue;

            float * const w1 = W + j1*align1 + f2*align0;
            float * const w2 = W + j2*align1 + f1*align0;

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
        }

        float * const w1 = W + j1*align1 + 39*align0;

        std::vector<float> sum_vec(nr_factor, 0);
        float * const sum = sum_vec.data();
        for(uint32_t f2 = 39; f2 < nr_field; ++f2)
        {
            uint32_t const j2 = J[f2];
            float * const w2 = W + j2*align1 + f1*align0;
            float * const wg2 = w2 + nr_factor; 

            for(uint32_t d = 0; d < nr_factor; d += 4)
            {
                __m128 XMMsum = _mm_load_ps(sum+d);
                __m128 XMMw1 = _mm_load_ps(w1+d);
                __m128 XMMw2 = _mm_load_ps(w2+d);
                XMMsum = _mm_add_ps(XMMsum, XMMw2);
                _mm_store_ps(sum+d, XMMsum);
                if(!do_update)
                    continue;
                __m128 XMMwg2 = _mm_load_ps(wg2+d);

                __m128 XMMg2 = _mm_add_ps(
                    _mm_mul_ps(XMMlambda, XMMw2),
                    _mm_mul_ps(XMMkappav, XMMw1));
                XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));
                XMMw2 = _mm_sub_ps(XMMw2,
                    _mm_mul_ps(XMMeta, 
                    _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));
                _mm_store_ps(w2+d, XMMw2);
                _mm_store_ps(wg2+d, XMMwg2);
            }
        }

        for(uint32_t d = 0; d < nr_factor; ++d)
            sum[d] /= 30;

        if(do_update)
        {
            float * const wg1 = w1 + nr_factor; 
            for(uint32_t d = 0; d < nr_factor; d += 4)
            {
                __m128 XMMsum = _mm_load_ps(sum+d);
                __m128 XMMw1 = _mm_load_ps(w1+d);
                __m128 XMMwg1 = _mm_load_ps(wg1+d);
                __m128 XMMg1 = _mm_add_ps(
                    _mm_mul_ps(XMMlambda, XMMw1),
                    _mm_mul_ps(XMMkappav, XMMsum));

                XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
                XMMw1 = _mm_sub_ps(XMMw1,
                    _mm_mul_ps(XMMeta, 
                    _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));

                _mm_store_ps(w1+d, XMMw1);

                _mm_store_ps(wg1+d, XMMwg1);
            }
        }
        else
        {
            for(uint32_t d = 0; d < nr_factor; d += 4)
            {
                __m128 const XMMsum = _mm_load_ps(sum+d);
                __m128 const XMMw1 = _mm_load_ps(w1+d);

                XMMt = _mm_add_ps(XMMt, _mm_mul_ps(_mm_mul_ps(XMMw1, XMMsum), XMMv));
            }
        }
    }

    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    float t;
    _mm_store_ss(&t, XMMt);

    return t;
}
*/

inline float wTx(SpMat const &spmat, Model &model, uint32_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    uint32_t const nr_factor = model.nr_factor;
    uint32_t const nr_field = model.nr_field;
    uint32_t const nr_feature = model.nr_feature;
    uint64_t const align0 = nr_factor*kW_NODE_SIZE;
    uint64_t const align1 = nr_field*align0;

    uint32_t const * const J = &spmat.J[i*nr_field];
    float * const W = model.W.data();
    float const v = spmat.v;

    float t = 0;
    for(uint32_t f1 = 0; f1 < 39; ++f1)
    {
        uint32_t const j1 = J[f1];
        if(j1 >= nr_feature)
            continue;

        for(uint32_t f2 = f1+1; f2 < 39; ++f2)
        {
            uint32_t const j2 = J[f2];
            if(j2 >= nr_feature)
                continue;

            float * const w1 = W + j1*align1 + f2*align0;
            float * const w2 = W + j2*align1 + f1*align0;

            if(do_update)
            {
                float * wg1 = w1 + nr_factor; 
                float * wg2 = w2 + nr_factor; 
                for(uint32_t d = 0; d < nr_factor; ++d)
                {
                    float const g1 = lambda*w1[d] + kappa*v*w2[d];
                    float const g2 = lambda*w2[d] + kappa*v*w1[d];

                    wg1[d] += g1*g1;
                    wg2[d] += g2*g2;

                    w1[d] -= eta*qrsqrt(wg1[d])*g1;
                    w2[d] -= eta*qrsqrt(wg2[d])*g2;
                }
            }
            else
            {
                for(uint32_t d = 0; d < nr_factor; ++d)
                    t += w1[d]*w2[d]*v;
            }
        }
    }

    std::vector<float> sav(nr_factor, 0), sbv(nr_factor, 0);
    float * const sa = sav.data(), * const sb = sbv.data();

    for(uint32_t f = 0; f < nr_field; ++f)
    {
        uint32_t const j = J[f];
        if(j >= nr_feature)
            continue;

        float * const w = W + j*align1 + 39*align0;
        float * const s = (f<39)? sa : sb;
        for(uint32_t d = 0; d < nr_factor; ++d)
                s[d] += w[d];
    }

    for(uint32_t d = 0; d < nr_factor; ++d)
    {
        sa[d] /= 39.0f;
        sb[d] /= 30.0f;
    }

    if(do_update)
    {
        for(uint32_t f = 0; f < nr_field; ++f)
        {
            uint32_t const j = J[f];
            if(j >= nr_feature)
                continue;

            float * const w = W + j*align1 + 39*align0;
            float * const wg = w + nr_factor; 
            float * const s = (f<39)? sa : sb;
            for(uint32_t d = 0; d < nr_factor; ++d)
            {
                float g = 0;
                g = lambda*w[d] + kappa*v*s[d];
                wg[d] += g*g;
                w[d] -= eta*qrsqrt(wg[d])*g;
            }
        }
    }
    else
    {
        for(uint32_t f = 0; f < nr_field; ++f)
        {
            uint32_t const j = J[f];
            if(j >= nr_feature)
                continue;

            float * const w = W + j*align1 + 39*align0;
            float * const s = (f<39)? sa : sb;
            for(uint32_t d = 0; d < nr_factor; ++d)
                t += w[d]*s[d]*v;
        }
    }

    return t;
}

float predict(SpMat const &spmat, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
