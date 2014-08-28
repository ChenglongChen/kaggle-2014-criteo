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
    Model(size_t const nr_feature, size_t const nr_factor) 
        : W(nr_feature*kNR_FIELD*nr_factor*kW_NODE_SIZE, 0), 
          nr_feature(nr_feature), nr_factor(nr_factor) {}
    std::vector<float> W;
    const size_t nr_feature, nr_factor;
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

    std::vector<float> cache(nr_factor, 0);
    float * const s = cache.data();

    for(size_t idx = spmat.P[i]; idx < spmat.P[i+1]; ++idx)
    {
        size_t const j = spmat.X[idx].j;
        float const v = spmat.X[idx].v;
        float * const w = model.W.data()+j*nr_factor*kW_NODE_SIZE;

        for(size_t d = 0; d < nr_factor; ++d)
            s[d] += w[d]*v;
    }

    float t = 0;
    for(size_t idx = spmat.P[i]; idx < spmat.P[i+1]; ++idx)
    {
        size_t const j = spmat.X[idx].j;
        float const v = spmat.X[idx].v;

        float * const w = model.W.data()+j*nr_factor*kW_NODE_SIZE;

        if(do_update)
        {
            float * const wg = w + nr_factor;
            for(size_t d = 0; d < nr_factor; ++d)
            {
                float const g = lambda*w[d] + kappa*v*(s[d]-w[d]*v);

                wg[d] += g*g;

                w[d] -= eta*qrsqrt(wg[d])*g;
            }
        }
        else
        {
            for(size_t d = 0; d < nr_factor; ++d)
            {
                float const vw = v*w[d];
                t += vw*(s[d]-vw);
            }
        }
    }

    return t/2.0f;
}

float predict(SpMat const &spmat, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
