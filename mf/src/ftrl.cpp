#include <cmath>

#include "common.h"
#include "ftrl.h"

void FTRL::load()
{
    FILE *f = open_c_file(model_path, "rb");

    fread(&param, sizeof(FTRLParameter), 1, f);

    size_t n;
    fread(&n, sizeof(size_t), 1, f);
    W.resize(n);
    Z.resize(n);
    N.resize(n);

    fread(W.data(), sizeof(double), n, f);
    fread(Z.data(), sizeof(double), n, f);
    fread(N.data(), sizeof(double), n, f);

    fclose(f);
}

void FTRL::save()
{
    FILE *f = open_c_file(model_path, "wb");

    fwrite(&param, sizeof(FTRLParameter), 1, f);

    size_t n = W.size();
    fwrite(&n, sizeof(size_t), 1, f);
    fwrite(W.data(), sizeof(double), n, f);
    fwrite(Z.data(), sizeof(double), n, f);
    fwrite(N.data(), sizeof(double), n, f);

    fclose(f);
}

void FTRL::update(int const y, std::vector<uint> const &idx, std::vector<double> const &val)
{
    double pred = 0;

    auto i = idx.begin();
    auto x = val.begin();
    for(; i != idx.end(); ++i, ++x)
    {
        if(*i >= W.size())
        {
            W.resize(*i+1, 0);
            Z.resize(*i+1, 0);
            N.resize(*i+1, 0);
        }
        auto w = W.begin()+(*i);
        auto z = Z.begin()+(*i);
        auto n = N.begin()+(*i);
        if(std::abs(*z) <= param.lambda1)
            *w = 0;
        else 
            *w = (-1/((param.beta+sqrt(*n))/param.alpha+param.lambda2))*
                ((*z)-((*z)>0?1:-1)*param.lambda1);
        pred += (*x)*(*w);
    }

    pred = 1/(1+exp(-pred));

    i = idx.begin();
    x = val.begin();
    for(; i != idx.end(); ++i, ++x)
    {
        auto w = W.begin()+(*i);
        auto z = Z.begin()+(*i);
        auto n = N.begin()+(*i);
        double g = (pred-static_cast<double>(y))*(*x);
        double sigma = (1/param.alpha)*(sqrt((*n)+g*g)-sqrt(*n));
        (*z) += g-sigma*(*w);
        (*n) += g*g;
    }
}
