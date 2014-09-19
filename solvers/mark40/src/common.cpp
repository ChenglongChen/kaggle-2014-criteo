#include <stdexcept>
#include <cstring>
#include <omp.h>

#include "common.h"

namespace {

inline float logistic_func(float const t)
{
    return 1/(1+static_cast<float>(exp(-t)));
}

uint32_t get_nr_line(std::string const &path)
{
    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    uint32_t nr_line = 0;
    while(fgets(line, kMaxLineSize, f) != nullptr)
        ++nr_line;

    fclose(f);

    return nr_line;
}

uint32_t get_nr_field(std::string const &path)
{
    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    fgets(line, kMaxLineSize, f);
    strtok(line, " \t");

    uint32_t nr_field = 0;
    while(1)
    {
        char *idx_char = strtok(nullptr," \t");
        if(idx_char == nullptr || *idx_char == '\n')
            break;
        ++nr_field;
    }

    fclose(f);

    return nr_field;
}

} //unamed namespace

std::vector<float> lambdas(39*39, 0.00002f);

SpMat read_data(std::string const path)
{
    if(path.empty())
        return SpMat(0, 0);
    SpMat spmat(get_nr_line(path), get_nr_field(path));

    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    uint64_t p = 0;
    for(uint32_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        char *y_char = strtok(line, " \t");
        float const y = (atoi(y_char)>0)? 1.0f : -1.0f;
        spmat.Y[i] = y;
        for(; ; ++p)
        {
            char *idx_char = strtok(nullptr," \t");
            if(idx_char == nullptr || *idx_char == '\n')
                break;
            uint32_t idx = static_cast<uint32_t>(atoi(idx_char));
            spmat.nr_feature = std::max(spmat.nr_feature, idx);
            spmat.J[p] = idx-1;
        }
    }

    fclose(f);

    return spmat;
}

FILE *open_c_file(std::string const &path, std::string const &mode)
{
    FILE *f = fopen(path.c_str(), mode.c_str());
    if(!f)
        throw std::runtime_error(std::string("cannot open ")+path);
    return f;
}

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}

float predict(SpMat const &spmat, Model &model, 
    std::string const &output_path)
{
    FILE *f = nullptr;
    if(!output_path.empty())
        f = open_c_file(output_path, "w");

    double loss = 0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for(uint32_t i = 0; i < spmat.Y.size(); ++i)
    {
        float const y = spmat.Y[i];

        float const t = wTx(spmat, model, i);
        
        float const prob = logistic_func(t);

        float const expnyt = static_cast<float>(exp(-y*t));

        loss += log(1+expnyt);

        if(f)
            fprintf(f, "%lf\n", prob);
    }

    if(f)
        fclose(f);

    return static_cast<float>(loss/static_cast<double>(spmat.Y.size()));
}
