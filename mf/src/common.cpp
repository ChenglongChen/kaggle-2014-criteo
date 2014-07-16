#include <stdexcept>
#include <cstring>

#include "common.h"

FILE *open_c_file(std::string const &path, std::string const &mode)
{
    FILE *f = fopen(path.c_str(), mode.c_str());
    if(!f)
        throw std::runtime_error(std::string("cannot open ")+path);
    return f;
}

void save_model(Model const &model, std::string const &path)
{
    FILE *f = open_c_file(path, "wb");
    fwrite(&model.n, sizeof(size_t), 1, f);
    fwrite(&model.k, sizeof(size_t), 1, f);
    fwrite(model.P.data(), sizeof(float), model.n*model.k, f);
    fclose(f);
}

Model read_model(std::string const &path)
{
    FILE *f = open_c_file(path, "rb");
    size_t n, k;
    fread(&n, sizeof(size_t), 1, f);
    fread(&k, sizeof(size_t), 1, f);
    Model model(n, k);
    fread(model.P.data(), sizeof(float), model.n*model.k, f);
    fclose(f);

    return model;
}

SpMat read_data(std::string const tr_path)
{
    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(tr_path.c_str(), "r");
    char line[kMaxLineSize];

    SpMat spmat;
    spmat.pv.push_back(0);
    while(fgets(line, kMaxLineSize, f) != nullptr)
    {
        char *p = strtok(line, " \t");
        int const y = (atoi(p)>0)? 1 : -1;
        while(1)
        {
            char *idx_char = strtok(nullptr,":");
            char *val_char = strtok(nullptr," \t");
            if(val_char == nullptr || *val_char == '\n')
                break;
            size_t idx = static_cast<size_t>(atoi(idx_char));
            spmat.n = std::max(spmat.n, idx);
            spmat.jv.push_back(idx-1);
        }
        spmat.pv.push_back(spmat.jv.size());
        spmat.yv.push_back(y);
    }

    fclose(f);

    return spmat;
}

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}
