#include <stdexcept>
#include <cstring>

#include "common.h"

namespace {

uint64_t const kMaxLineSize = 1000000;

uint64_t get_nr_line(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    uint64_t nr_line = 0;
    while(fgets(line, kMaxLineSize, f) != nullptr)
        ++nr_line;

    fclose(f);

    return nr_line;
}

uint64_t get_nr_field(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    fgets(line, kMaxLineSize, f);
    strtok(line, " \t");

    uint64_t nr_field = 0;
    while(1)
    {
        char *val_char = strtok(nullptr," \t");
        if(val_char == nullptr || *val_char == '\n')
            break;
        ++nr_field;
    }

    fclose(f);

    return nr_field;
}

} //unamed namespace

DenseColMat read_dcm(std::string const &path)
{
    if(path.empty())
        return DenseColMat(0, 0);

    DenseColMat problem(get_nr_line(path), get_nr_field(path));

    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");
    for(uint64_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        char *str_ptr = strtok(line, " \t");
        problem.Y[i] = (atoi(str_ptr)>0)? 1.0f : -1.0f;
        for(uint64_t j = 0; j < problem.nr_field; ++j)
        {
            char *val_char = strtok(nullptr," \t");

            float const val = static_cast<float>(atof(val_char));

            problem.X[j][i] = val;
        }
    }

    fclose(f);

    return problem;
}

SparseColMat read_scm(std::string const &path)
{
    if(path.empty())
        return SparseColMat(0, 0, 0);

    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");

    std::vector<std::vector<uint64_t>> buffer;

    uint64_t nnz = 0, nr_instance = 0;
    for(uint64_t i = 0; fgets(line, kMaxLineSize, f) != nullptr;
        ++i, ++nr_instance)
    {
        strtok(line, " \t");
        for( ; ; ++nnz)
        {
            char *idx_char = strtok(nullptr," \t");

            uint64_t const idx = atol(idx_char);
            if(idx_char == nullptr || *idx_char == '\n')
                break;

            buffer.resize(idx);

            buffer[idx-1].push_back(i);
        }
    }

    uint64_t const nr_field = buffer.size();

    SparseColMat problem(nr_instance, nr_field, nnz);

    problem.P[0] = 0;

    uint64_t p = 0;
    for(uint64_t j = 0; j < nr_field; ++j)
    {
        for(auto i : buffer[j]) 
            problem.X[p++] = i;
        problem.P[j+1] = p;
    }

    fclose(f);

    return problem;
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