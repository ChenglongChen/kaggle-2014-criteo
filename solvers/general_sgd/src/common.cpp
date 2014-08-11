#include <stdexcept>
#include <cstring>

#include "common.h"

SpMat read_data(std::string const tr_path)
{
    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(tr_path.c_str(), "r");
    char line[kMaxLineSize];

    SpMat spmat;
    spmat.P.push_back(0);
    while(fgets(line, kMaxLineSize, f) != nullptr)
    {
        char *p = strtok(line, " \t");
        int const y = (atoi(p)>0)? 1 : 0;
        while(1)
        {
            char *idx_char = strtok(nullptr,":");
            char *val_char = strtok(nullptr," \t");
            if(val_char == nullptr || *val_char == '\n')
                break;
            size_t idx = static_cast<size_t>(atoi(idx_char));
            spmat.n = std::max(spmat.n, idx);
            spmat.J.push_back(idx-1);
        }
        spmat.P.push_back(spmat.J.size());
        spmat.Y.push_back(y);
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
