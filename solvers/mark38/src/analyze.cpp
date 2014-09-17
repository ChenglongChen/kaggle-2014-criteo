#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

#include "common.h"

namespace {

struct Option
{
    std::string Te_path, model_path, output_path;
};

std::string train_help()
{
    return std::string(
"usage: mark38-predict <test_path> <model_path>\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    size_t const argc = args.size();

    if(argc != 2)
        throw std::invalid_argument(train_help());

    Option option; 
    option.Te_path = args[0];
    option.model_path = args[1];

    return option;
}

void analyze_acc(SpMat const &spmat, Model &model, 
    std::string const &output_path)
{
    struct Info
    {
        Info() : correct(0), total(0) {}
        uint32_t correct, total;
    };

    uint32_t const nr_factor = model.nr_factor;
    uint32_t const nr_field = model.nr_field;
    uint32_t const nr_feature = model.nr_feature;
    uint32_t const nr_instance = spmat.nr_instance;
    uint64_t const align0 = nr_factor*kW_NODE_SIZE;
    uint64_t const align1 = nr_field*align0;

    std::vector<Info> records(nr_field*nr_field);
    for(uint32_t i = 0; i < spmat.Y.size(); ++i)
    {
        float const y = spmat.Y[i];

        for(uint32_t f1 = 0; f1 < nr_field; ++f1)
        {
            uint32_t const j1 = spmat.J[i*spmat.nr_field+f1];
            if(j1 >= nr_feature)
                continue;

            for(uint32_t f2 = f1+1; f2 < nr_field; ++f2)
            {
                uint32_t const j2 = spmat.J[i*spmat.nr_field+f2];
                if(j2 >= nr_feature)
                    continue;

                float * const w1 = model.W.data() + j1*align1 + f2*align0;
                float * const w2 = model.W.data() + j2*align1 + f1*align0;

                float t1 = 0;
                for(uint32_t d = 0; d < nr_factor; ++d)
                    t1 += w1[d]*w2[d]*spmat.v;

                if(y * t1 > 0)
                    ++records[f1*nr_field+f2].correct;
            }
        }
    }

    for(uint32_t f1 = 0; f1 < nr_field; ++f1)
    {
        for(uint32_t f2 = f1+1; f2 < nr_field; ++f2)
        {
            float const acc = 
                static_cast<float>(records[f1*nr_field+f2].correct) / 
                static_cast<float>(nr_instance);
            printf("%3d %3d %.3f\n", f1, f2, acc);
            fflush(stdout);
        }
    }

}

} //unnamed namespace

int main(int const argc, char const * const * const argv)
{
    Option opt;
    try
    {
        opt = parse_option(argv_to_args(argc, argv));
    }
    catch(std::invalid_argument const &e)
    {
        std::cout << "\n" << e.what() << "\n";
        return EXIT_FAILURE;
    }

	omp_set_num_threads(1);

    SpMat const Te = read_data(opt.Te_path);

    Model model = load_model(opt.model_path);
    
    //float const Te_loss = predict(Te, model, opt.output_path);

    //printf("logloss = %f\n", Te_loss);

    analyze_acc(Te, model, opt.output_path);

    return EXIT_SUCCESS;
}
