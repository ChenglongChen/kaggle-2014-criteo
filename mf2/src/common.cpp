#include <stdexcept>
#include <cstring>

#include "common.h"

std::vector<size_t> const FieldSizes = {};
std::vector<size_t> const A = {1, 2, 3};
std::vector<size_t> const B = {1, 2, 3};

SpMat read_data(std::string const tr_path)
{
    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(tr_path.c_str(), "r");
    char line[kMaxLineSize];

    SpMat spmat;
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
            size_t const idx = static_cast<size_t>(atoi(idx_char));
            spmat.X.push_back(idx-1);
        }
        spmat.Y.push_back(y);
    }

    fclose(f);

    return spmat;
}

void save_model(Model const &model, std::string const &path)
{
    //FILE *f = open_c_file(path, "wb");
    //fwrite(&model.n, sizeof(size_t), 1, f);
    //fwrite(&model.k, sizeof(size_t), 1, f);
    //fwrite(model.P.data(), sizeof(float), model.n*model.k, f);
    //fwrite(model.W.data(), sizeof(float), model.n, f);
    //fclose(f);
}

Model read_model(std::string const &path)
{
    //FILE *f = open_c_file(path, "rb");
    //size_t n, k;
    //fread(&n, sizeof(size_t), 1, f);
    //fread(&k, sizeof(size_t), 1, f);
    //Model model(n, k);
    //fread(model.P.data(), sizeof(float), model.n*model.k, f);
    //fread(model.W.data(), sizeof(float), model.n, f);
    //fclose(f);

    return Model(0, FieldSizes);
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
