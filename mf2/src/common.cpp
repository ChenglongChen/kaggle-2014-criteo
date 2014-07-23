#include <stdexcept>
#include <cstring>

#include "common.h"

std::vector<size_t> const FieldSizes = 
//   I1       I2       I3       I4       I5       I6       I7       I8       I9       I10      I11      I12      I13
    {100000,  100000,  100000,  100000,  1000000, 100000,  100000,  100000,  100000,  100000,  100000,  100000,  100000,
//   C1       C2       C3       C4       C5       C6       C7       C8       C9       C10      C11      C12      C13      C14      C15      C16      C17      C18      C19      C20      C21      C22      C23      C24      C25      C26
     100000,  100000,  1000000, 1000000, 100000,  100000,  1000000, 100000,  100000,  1000000, 100000,  1000000, 100000,  100000,  1000000, 1000000, 100000,  100000,  100000,  100000,  1000000, 100000,  100000,  1000000, 100000,  1000000};
std::vector<size_t> const A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38};
std::vector<size_t> const B = {13, 14, 17, 18, 20, 21, 23, 25, 26, 29, 30, 31, 32, 34, 35, 37};

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
            spmat.X.push_back(idx);
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
