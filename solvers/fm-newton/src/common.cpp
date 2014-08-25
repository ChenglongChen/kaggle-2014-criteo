#include <stdexcept>
#include <cstring>
#include <omp.h>

#include "common.h"

namespace {

inline double logistic_func(double const t)
{
    return 1/(1+static_cast<double>(exp(-t)));
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
        double const y = (atoi(p)>0)? 1.0f : -1.0f;
        while(1)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *val_char = strtok(nullptr," \t");
            if(val_char == nullptr || *val_char == '\n')
                break;
            size_t field = static_cast<size_t>(atoi(field_char));
            size_t idx = static_cast<size_t>(atoi(idx_char));
            double const val = static_cast<double>(atof(val_char));
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

void save_model(Model const &model, std::string const &path)
{
    FILE *f = fopen(path.c_str(), "wb");
    fwrite(&model.nr_feature, sizeof(size_t), 1, f);
    fwrite(&model.nr_factor, sizeof(size_t), 1, f);
    fwrite(model.W.data(), sizeof(double), 
        model.nr_feature*kNR_FIELD*model.nr_factor, f);
    fclose(f);
}

Model load_model(std::string const &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    size_t nr_factor, nr_feature;
    fread(&nr_feature, sizeof(size_t), 1, f);
    fread(&nr_factor, sizeof(size_t), 1, f);

    Model model(nr_feature, nr_factor);
    fread(model.W.data(), sizeof(double), 
        model.nr_feature*kNR_FIELD*model.nr_factor, f);
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

double predict(SpMat const &problem, Model &model, 
    std::string const &output_path)
{
    FILE *f = nullptr;
    if(!output_path.empty())
        f = open_c_file(output_path, "w");

    double loss = 0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for(size_t i = 0; i < problem.Y.size(); ++i)
    {
        double const y = problem.Y[i];

        double const t = wTx(problem, model, i);
        
        double const prob = logistic_func(t);

        double const expnyt = static_cast<double>(exp(-y*t));

        loss += log(1+expnyt);

        if(f)
            fprintf(f, "%lf\n", prob);
    }

    if(f)
        fclose(f);

    return static_cast<double>(loss/static_cast<double>(problem.Y.size()));
}
