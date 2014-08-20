#include <stdexcept>
#include <cstring>
#include <omp.h>

#include "common.h"

namespace {

inline float logistic_func(float const t)
{
    return 1/(1+static_cast<float>(exp(-t)));
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
            spmat.n = std::max(spmat.n, idx);
            spmat.JX.emplace_back(field-1, idx-1, val);
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
    fwrite(&model.n, sizeof(size_t), 1, f);
    fwrite(&model.k, sizeof(size_t), 1, f);
    fwrite(model.W.data(), sizeof(float), 
        model.n*kF_SIZE*model.k*kW_NODE_SIZE, f);
    fclose(f);
}

Model load_model(std::string const &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    size_t k, n;
    fread(&n, sizeof(size_t), 1, f);
    fread(&k, sizeof(size_t), 1, f);

    Model model(n, k);
    fread(model.W.data(), sizeof(float), 
        model.n*kF_SIZE*model.k*kW_NODE_SIZE, f);
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

float predict(SpMat const &problem, Model &model, 
    std::string const &output_path, bool const analyze)
{
    FILE *f = nullptr;
    if(!output_path.empty())
        f = open_c_file(output_path, "w");

    double loss = 0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for(size_t i = 0; i < problem.Y.size(); ++i)
    {
        float y = problem.Y[i];

        float t = wTx(problem, model, i);
        
        float prob = logistic_func(t);

        //if((prob > 0.8 && y == 0) || (prob < 0.2 && y == 1))
        //    printf("%d %f\n", static_cast<int>(y), prob);

        if(prob > 0.9)
        {
            if(true)
            {
                printf("%2d %f\n", static_cast<int>(y), prob);
                size_t const k = model.k;
                float t_ = 0;
                for(size_t idx1 = problem.P[i]; idx1 < problem.P[i+1]; ++idx1)
                {
                    size_t const j1 = problem.JX[idx1].j;
                    size_t const f1 = problem.JX[idx1].f;
                    float const x1 = problem.JX[idx1].x;

                    for(size_t idx2 = idx1+1; idx2 < problem.P[i+1]; ++idx2)
                    {
                        size_t const j2 = problem.JX[idx2].j;
                        size_t const f2 = problem.JX[idx2].f;
                        float const x2 = problem.JX[idx2].x;

                        float * const w1 = 
                            model.W.data()+j1*kF_SIZE*k*kW_NODE_SIZE+f2*k*kW_NODE_SIZE;
                        float * const w2 = 
                            model.W.data()+j2*kF_SIZE*k*kW_NODE_SIZE+f1*k*kW_NODE_SIZE;
                        
                        float t1 = 0;
                        for(size_t d = 0; d < k; d += 1)
                            t1 += (*(w1+d))*(*(w2+d))*x1*x2;
                        t_ += t1;
                        if(t1 > 0.1)
                            printf("f1 = %3ld, f2 = %3ld, j1 = %10ld, j2 = %10ld, w1 = %10.3f, w2 = %10.3f, t1 = %10.3f, t = %10.3f\n", f1, f2, j1, j2, *w1, *w2, t1, t_);
                    }
                }
            }
        }

        float const expnyt = static_cast<float>(exp(-y*t));

        loss += log(1+expnyt);

        if(f)
            fprintf(f, "%lf\n", prob);
    }

    if(f)
        fclose(f);

    return static_cast<float>(loss/static_cast<double>(problem.Y.size()));
}
