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
    Node(size_t const j, float const v) : j(j), v(v) {}
    size_t j;
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

SpMat read_data(std::string const path, size_t const reserved_size=0);

struct WNode
{
    WNode() : v(0), sg2(1) {}
    float v, sg2;
};

struct Model
{
    Model(size_t const nr_feature) : W(nr_feature), nr_feature(nr_feature) {}
    std::vector<WNode> W;
    const size_t nr_feature;
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
    float t = 0;
    for(size_t p = spmat.P[i]; p < spmat.P[i+1]; ++p)
    {
        Node const &x = spmat.X[p];
        WNode &w = model.W[x.j];

        if(do_update)
        {
            float const g = lambda*w.v + kappa*x.v;

            w.sg2 += g*g;

            w.v -= eta*qrsqrt(w.sg2)*g;
        }
        else
        {
            t += w.v*x.v;
        }
    }

    return t;
}

float predict(SpMat const &spmat, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
