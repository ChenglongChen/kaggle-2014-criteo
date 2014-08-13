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
        float const y = (atoi(p)>0)? 1.0f : -1.0f;
        while(1)
        {
            char *idx_char = strtok(nullptr,":");
            char *val_char = strtok(nullptr," \t");
            if(val_char == nullptr || *val_char == '\n')
                break;
            size_t idx = static_cast<size_t>(atoi(idx_char));
            float const val = static_cast<float>(atof(val_char));
            spmat.n = std::max(spmat.n, idx);
            spmat.JX.emplace_back(idx-1, val);
        }
        spmat.P.push_back(spmat.JX.size());
        spmat.Y.push_back(y);
    }

    fclose(f);

    return spmat;
}

void save_model(Model const &model, std::string const &path)
{
    FILE *f = fopen(path.c_str(), "wb");
    fwrite(model.W.data(), sizeof(WNode),kW_SIZE, f);
    fclose(f);
}

Model load_model(std::string const &path)
{
    Model model;
    FILE *f = fopen(path.c_str(), "rb");
    fread(model.W.data(), sizeof(WNode), kW_SIZE, f);
    fclose(f);
    return model;
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

float predict(SpMat const &problem, Model const &model, 
    std::string const &output_path)
{
    FILE *f = nullptr;
    if(!output_path.empty())
        f = open_c_file(output_path, "w");

    double loss = 0;
    for(size_t i = 0; i < problem.Y.size(); ++i)
    {
        float const y = problem.Y[i];

        float const t = wTx(problem, model, i);
        
        float const prob = logistic_func(t);

        float const expnyt = static_cast<float>(exp(-y*t));

        loss += log(1+expnyt);

        if(f)
            fprintf(f, "%lf\n", prob);
    }

    if(f)
        fclose(f);

    return static_cast<float>(loss/static_cast<double>(problem.Y.size()));
}
