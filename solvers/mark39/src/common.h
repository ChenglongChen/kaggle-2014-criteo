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

struct Model
{
    Model(uint32_t const nr_feature, uint32_t const nr_factor, uint32_t const nr_field) 
        : W(static_cast<uint64_t>(nr_feature)*nr_field*nr_factor*kW_NODE_SIZE, 0), 
          G(static_cast<uint64_t>(nr_feature)*nr_field*nr_factor*kW_NODE_SIZE, 0), 
          nr_feature(nr_feature), nr_factor(nr_factor), nr_field(nr_field) {}
    std::vector<float> W, G;
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
    float * const G = model.G.data();
    float const v = spmat.v;

    float t = 0;
    for(uint32_t f1 = 0; f1 < nr_field; ++f1)
    {
        uint32_t const j1 = J[f1];
        if(j1 >= nr_feature)
            continue;

        for(uint32_t f2 = f1+1; f2 < nr_field; ++f2)
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

                float * const g1 = G + j1*align1 + f2*align0;
                float * const g2 = G + j2*align1 + f1*align0;

                for(uint32_t d = 0; d < nr_factor; ++d)
                {
                    g1[d] = kappa*v*w2[d];
                    g2[d] = kappa*v*w1[d];

                    float const g1_ = lambda*w1[d] + g1[d];
                    float const g2_ = lambda*w2[d] + g2[d];

                    wg1[d] += g1_*g1_;
                    wg2[d] += g2_*g2_;

                    w1[d] -= eta*qrsqrt(wg1[d])*g1_;
                    w2[d] -= eta*qrsqrt(wg2[d])*g2_;
                }
            }
            else
            {
                for(uint32_t d = 0; d < nr_factor; ++d)
                    t += w1[d]*w2[d]*v;
            }
        }
    }

    return t;
}

float predict(SpMat const &spmat, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
