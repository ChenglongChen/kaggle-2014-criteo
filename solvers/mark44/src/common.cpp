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

Problem read_dcm(std::string const &path)
{
    if(path.empty())
        return Problem(0, 0);

    Problem problem(get_nr_line(path), get_nr_field(path));

    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");
    for(uint64_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        char *p = strtok(line, " \t");
        problem.Y[i] = (atoi(p)>0)? 1.0f : -1.0f;
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
