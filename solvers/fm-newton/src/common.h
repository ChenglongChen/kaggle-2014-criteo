#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

struct Node
{
    Node(size_t const f, size_t const j, double const v) : f(f), j(j), v(v) {}
    size_t f, j;
    double v;
};

struct SpMat
{
    SpMat() : nr_feature(0), nr_instance(0) {}
    std::vector<size_t> P;
    std::vector<Node> X;
    std::vector<double> Y;
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
    std::vector<double> W;
    const size_t nr_feature, nr_factor;
};

void save_model(Model const &model, std::string const &path);

Model load_model(std::string const &path);

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline double wTx(SpMat const &spmat, Model const &model, size_t const i,
    double const * W=nullptr)
{
    size_t const nr_factor = model.nr_factor;
    if(W == nullptr)
        W = model.W.data();

    double t = 0;
    for(size_t idx1 = spmat.P[i]; idx1 < spmat.P[i+1]; ++idx1)
    {
        size_t const j1 = spmat.X[idx1].j;
        size_t const f1 = spmat.X[idx1].f;
        double const v1 = spmat.X[idx1].v;

        for(size_t idx2 = idx1+1; idx2 < spmat.P[i+1]; ++idx2)
        {
            size_t const j2 = spmat.X[idx2].j;
            size_t const f2 = spmat.X[idx2].f;
            double const v2 = spmat.X[idx2].v;

            double const * w1 = 
                W+j1*kNR_FIELD*nr_factor*kW_NODE_SIZE+f2*nr_factor*kW_NODE_SIZE;
            double const * w2 = 
                W+j2*kNR_FIELD*nr_factor*kW_NODE_SIZE+f1*nr_factor*kW_NODE_SIZE;

            for(size_t d = 0; d < nr_factor; ++d, ++w1, ++w2)
                t += (*w1)*(*w2)*v1*v2;
        }
    }

    return t;
}

double predict(SpMat const &spmat, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
