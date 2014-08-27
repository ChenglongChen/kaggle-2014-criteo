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

            for(size_t idx3 = idx2+1; idx3 < spmat.P[i+1]; ++idx3)
            {
                size_t const j3 = spmat.X[idx3].j;
                size_t const f3 = spmat.X[idx3].f;
                float const v3 = spmat.X[idx3].v;

                float * w12 = model.W.data()+j1*kNR_FIELD*nr_factor*kW_NODE_SIZE+f2*nr_factor*kW_NODE_SIZE;
                float * w13 = model.W.data()+j1*kNR_FIELD*nr_factor*kW_NODE_SIZE+f3*nr_factor*kW_NODE_SIZE;
                float * w21 = model.W.data()+j2*kNR_FIELD*nr_factor*kW_NODE_SIZE+f1*nr_factor*kW_NODE_SIZE;
                float * w23 = model.W.data()+j2*kNR_FIELD*nr_factor*kW_NODE_SIZE+f3*nr_factor*kW_NODE_SIZE;
                float * w31 = model.W.data()+j3*kNR_FIELD*nr_factor*kW_NODE_SIZE+f1*nr_factor*kW_NODE_SIZE;
                float * w32 = model.W.data()+j3*kNR_FIELD*nr_factor*kW_NODE_SIZE+f2*nr_factor*kW_NODE_SIZE;

                if(do_update)
                {
                    float * wg12 = w12 + nr_factor; 
                    float * wg13 = w13 + nr_factor; 
                    float * wg21 = w21 + nr_factor; 
                    float * wg23 = w23 + nr_factor; 
                    float * wg31 = w31 + nr_factor; 
                    float * wg32 = w32 + nr_factor; 
                    for(size_t d = 0; d < nr_factor; ++d, ++w12, ++w13, ++w21, ++w23, ++w31, ++w32, ++wg12, ++wg13, ++wg21, ++wg23, ++wg31, ++wg32)
                    {
                        float const wall = (*w12)*(*w13)*(*w21)*(*w23)*(*w31)*(*w32);

                        float const g12 = lambda*(*w12) + kappa*v1*v2*v3*wall/(*w12);
                        float const g13 = lambda*(*w13) + kappa*v1*v2*v3*wall/(*w13);
                        float const g21 = lambda*(*w21) + kappa*v1*v2*v3*wall/(*w21);
                        float const g23 = lambda*(*w23) + kappa*v1*v2*v3*wall/(*w23);
                        float const g31 = lambda*(*w31) + kappa*v1*v2*v3*wall/(*w31);
                        float const g32 = lambda*(*w32) + kappa*v1*v2*v3*wall/(*w32);

                        *wg12 += g12*g12;
                        *wg13 += g13*g13;
                        *wg21 += g21*g21;
                        *wg23 += g23*g23;
                        *wg31 += g31*g31;
                        *wg32 += g32*g32;

                        *w12 -= eta*qrsqrt(*wg12)*g12;
                        *w13 -= eta*qrsqrt(*wg13)*g13;
                        *w21 -= eta*qrsqrt(*wg21)*g21;
                        *w23 -= eta*qrsqrt(*wg23)*g23;
                        *w31 -= eta*qrsqrt(*wg31)*g31;
                        *w32 -= eta*qrsqrt(*wg32)*g32;
                    }
                }
                else
                {
                    for(size_t d = 0; d < nr_factor; ++d, ++w12, ++w13, ++w21, ++w23, ++w31, ++w32)
                        t += (*w12)*(*w13)*(*w21)*(*w23)*(*w31)*(*w32)*v1*v2*v3;
                }
            }
        }
    }

    return t;
}

float predict(SpMat const &spmat, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
