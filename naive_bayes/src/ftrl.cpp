#include <cmath>

#include "common.h"
#include "ftrl.h"

void FTRL::load()
{
    FILE *f = open_c_file(model_path, "rb");

    fread(&pos, sizeof(long long), 1, f);
    fread(&neg, sizeof(long long), 1, f);

    size_t n;
    fread(&n, sizeof(size_t), 1, f);
    likelihood.resize(n);
    evidence.resize(n);

    fread(likelihood.data(), sizeof(long long), n, f);
    fread(evidence.data(), sizeof(long long), n, f);

    fclose(f);
}

void FTRL::save()
{
    FILE *f = open_c_file(model_path, "wb");

    fwrite(&pos, sizeof(long long), 1, f);
    fwrite(&neg, sizeof(long long), 1, f);

    size_t n = likelihood.size();
    fwrite(&n, sizeof(size_t), 1, f);
    fwrite(likelihood.data(), sizeof(long long), n, f);
    fwrite(evidence.data(), sizeof(long long), n, f);

    fclose(f);
}

void FTRL::update(int const y, std::vector<uint> const &idx, std::vector<double> const &val)
{
    if(y == 1)
        ++pos;
    else
        ++neg;

    for(auto i = idx.begin(); i != idx.end(); ++i)
    {
        if(*i >= likelihood.size())
        {
            likelihood.resize(*i+1, 0);
            evidence.resize(*i+1, 0);
        }
        if(y == 1)
            ++likelihood[*i];
        ++evidence[*i];
    }
}
