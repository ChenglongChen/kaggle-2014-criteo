#include <stdexcept>
#include <cstring>

#include "common.h"

namespace {

inline float logistic_func(float const t)
{
    return 1/(1+static_cast<float>(exp(-t)));
}

} //unamed namespace

SpMat read_data(std::string const path, size_t const reserved_size)
{
    SpMat spmat;
    if(path.empty())
        return spmat;

    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    spmat.P.push_back(0);
    spmat.X.reserve(reserved_size);
    while(fgets(line, kMaxLineSize, f) != nullptr)
    {
        char *p = strtok(line, " \t");
        float const y = (atoi(p)>0)? 1.0f : -1.0f;
        while(1)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *val_char = strtok(nullptr," \t");
            if(val_char == nullptr || *val_char == '\n')
                break;
            size_t field = static_cast<size_t>(atoi(field_char));
            size_t idx = static_cast<size_t>(atoi(idx_char));
            float const val = static_cast<float>(atof(val_char));
            spmat.nr_feature = std::max(spmat.nr_feature, idx);
            spmat.X.emplace_back(field-1, idx-1, val);
        }
        spmat.P.push_back(spmat.X.size());
        spmat.Y.push_back(y);
        ++spmat.nr_instance;
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
