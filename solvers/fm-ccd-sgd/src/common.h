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
    SpMat() : nr_instance(0), nr_field(0) {}
    std::vector<size_t> P;
    std::vector<Node> X;
    std::vector<float> Y;
    std::vector<size_t> nr_field_feature;
    size_t nr_instance, nr_field;
};

SpMat read_data(std::string const path);

struct Model
{
    Model(size_t const nr_field, size_t const nr_factor, 
          std::vector<size_t> nr_field_feature)
        : nr_field(nr_field), nr_factor(nr_factor), 
          nr_field_feature(nr_field_feature), W(nr_field), WG(nr_field)
    {
        for(size_t f = 0; f < nr_field; ++f) 
        {
            W[f].resize(nr_field_feature[f]*nr_field*nr_factor);
            WG[f].resize(nr_field_feature[f]*nr_field*nr_factor, 1);
        }
    }
    size_t const nr_field, nr_factor;
    std::vector<size_t> const nr_field_feature; 
    std::vector<std::vector<float>> W, WG;
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

inline float phi(SpMat const &spmat, Model const &model, size_t const i)
{
    size_t const nr_field = model.nr_field;
    size_t const nr_factor = model.nr_factor;

    float t = 0;
    for(size_t idx1 = spmat.P[i]; idx1 < spmat.P[i+1]; ++idx1)
    {
        size_t const j1 = spmat.X[idx1].j;
        size_t const f1 = spmat.X[idx1].f;
        if(j1 >= model.nr_field_feature[f1])
            continue;
        float const v1 = spmat.X[idx1].v;

        for(size_t idx2 = idx1+1; idx2 < spmat.P[i+1]; ++idx2)
        {
            size_t const j2 = spmat.X[idx2].j;
            size_t const f2 = spmat.X[idx2].f;
            if(j2 >= model.nr_field_feature[f2])
                continue;
            float const v2 = spmat.X[idx2].v;

            float const * w1 = 
                model.W[f1].data()+j1*nr_field*nr_factor+f2*nr_factor;
            float const * w2 = 
                model.W[f2].data()+j2*nr_field*nr_factor+f1*nr_factor;

            for(size_t d = 0; d < nr_factor; ++d, ++w1, ++w2)
                t += v1*v2*(*w1)*(*w2);
        }
    }

    return t;
}

inline float wTx(SpMat const &problem, Model &model, size_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    size_t const nr_field = model.nr_field;
    size_t const nr_factor = model.nr_factor;

    float t = 0;
    for(size_t idx1 = problem.P[i]; idx1 < problem.P[i+1]; ++idx1)
    {
        size_t const j1 = problem.X[idx1].j;
        size_t const f1 = problem.X[idx1].f;
        float const v1 = problem.X[idx1].v;

        for(size_t idx2 = idx1+1; idx2 < problem.P[i+1]; ++idx2)
        {
            size_t const j2 = problem.X[idx2].j;
            size_t const f2 = problem.X[idx2].f;
            float const v2 = problem.X[idx2].v;

            float * w1 = 
                model.W[f1].data()+j1*nr_field*nr_factor+f2*nr_factor;
            float * wg1 = 
                model.WG[f1].data()+j1*nr_field*nr_factor+f2*nr_factor;
            float * w2 = 
                model.W[f2].data()+j2*nr_field*nr_factor+f1*nr_factor;
            float * wg2 = 
                model.WG[f2].data()+j2*nr_field*nr_factor+f1*nr_factor;

            if(do_update)
            {
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
                    t += v1*v2*(*w1)*(*w2);
            }
        }
    }

    return t;
}

std::vector<float> calc_s(SpMat const &spmat, Model const &model);

float calc_loss(std::vector<float> const &Y, std::vector<float> const &S,
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
