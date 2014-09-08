#include <stdexcept>
#include <cstring>

#include "common.h"

namespace {

size_t get_nr_line(std::string const &path)
{
    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    size_t nr_line = 0;
    while(fgets(line, kMaxLineSize, f) != nullptr)
        ++nr_line;

    fclose(f);

    return nr_line;
}

} //unamed namespace

Problem read_data(std::string const &path)
{
    if(path.empty())
        return Problem(0);

    Problem problem(get_nr_line(path));

    int const kMaxLineSize = 1000000;

    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");
    for(size_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        char *p = strtok(line, " \t");
        problem.Y[i] = (atoi(p)>0)? 1.0f : -1.0f;
        for(size_t j = 0; j < kNR_FEATURE; ++j)
        {
            strtok(nullptr,":");
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
