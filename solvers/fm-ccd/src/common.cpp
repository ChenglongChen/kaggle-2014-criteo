#include <stdexcept>
#include <cstring>
#include <omp.h>

#include "common.h"

namespace {

inline float logistic_func(float const s)
{
    return 1/(1+static_cast<float>(exp(-s)));
}

} //unamed namespace

SpMat read_data(std::string const path)
{
    SpMat spmat;
    if(path.empty())
        return spmat;

    int const kMaxLineSize = 1000000;
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    spmat.P.push_back(0);
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

            spmat.nr_field = std::max(spmat.nr_field, field);
            if(spmat.nr_field_feature.size() != spmat.nr_field)
            {
                if(spmat.nr_instance != 0)
                    throw std::runtime_error(
                        "number of fields in each line shoud be the same");
                spmat.nr_field_feature.resize(spmat.nr_field, 0);
            }

            spmat.nr_field_feature[field-1] 
                = std::max(spmat.nr_field_feature[field-1], idx);
            spmat.X.emplace_back(field-1, idx-1, val);
        }
        spmat.P.push_back(spmat.X.size());
        spmat.Y.push_back(y);
        ++spmat.nr_instance;
    }

    fclose(f);

    return spmat;
}

void save_model(Model const &model, std::string const &path)
{
    FILE *fout = fopen(path.c_str(), "wb");
    fwrite(&model.nr_field, sizeof(size_t), 1, fout);
    fwrite(&model.nr_factor, sizeof(size_t), 1, fout);
    fwrite(model.nr_field_feature.data(), sizeof(size_t), model.nr_field, fout);
    for(size_t f = 0; f < model.nr_field; ++f)
        fwrite(model.W[f].data(), sizeof(float), model.nr_field_feature[f] * 
            model.nr_field * model.nr_factor, fout);
    fclose(fout);
}

Model load_model(std::string const &path)
{
    FILE *fin = fopen(path.c_str(), "rb");
    size_t nr_field, nr_factor;
    fread(&nr_field, sizeof(size_t), 1, fin);
    fread(&nr_factor, sizeof(size_t), 1, fin);

    std::vector<size_t> nr_field_feature(nr_field);
    fread(nr_field_feature.data(), sizeof(size_t), nr_field, fin);

    Model model(nr_field, nr_factor, nr_field_feature);
    for(size_t f = 0; f < model.nr_field; ++f)
        fread(model.W[f].data(), sizeof(float), model.nr_field_feature[f] * 
            model.nr_field * model.nr_factor, fin);
    fclose(fin);
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

std::vector<float> calc_s(SpMat const &spmat, Model const &model)
{
    std::vector<float> S(spmat.nr_instance);
    for(size_t i = 0; i < spmat.nr_instance; ++i)
        S[i] = phi(spmat, model, i);
    return S;
}

float calc_loss(std::vector<float> const &Y, std::vector<float> const &S,
    std::string const &output_path)
{
    FILE *f = nullptr;
    if(!output_path.empty())
        f = open_c_file(output_path, "w");

    double loss = 0;
    for(size_t i = 0; i < Y.size(); ++i)
    {
        float const y = Y[i];
        
        float const s = S[i];

        float const expnyt = static_cast<float>(exp(-y*s));

        loss += log(1+expnyt);

        if(f)
        {
            float const prob = logistic_func(s);
            fprintf(f, "%lf\n", prob);
        }
    }

    if(f)
        fclose(f);

    return static_cast<float>(loss/static_cast<double>(Y.size()));
}
