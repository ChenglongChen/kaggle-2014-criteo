#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include "common.h"

namespace {

struct Option 
{
    std::string te_path, model_path, output_path;
};

std::string train_help()
{
    return std::string(
"usage: online-predict test_path model_path output_path\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    size_t const argc = args.size();

    Option opt;
    if(argc != 3)
    {
        throw std::invalid_argument(train_help());
    }
    else
    {
        opt.te_path = args[0];
        opt.model_path = args[1];
        opt.output_path = args[2];
    }

    return opt;
}

void 
predict(SpMat const &Te, Model const &model, std::string const &output_path)
{
    FILE *f = open_c_file(output_path, "w");     

    for(size_t i = 0; i < Te.pv.size()-1; ++i)
    {
        size_t nnz = Te.pv[i+1]-Te.pv[i];
        if(nnz <= 1)
        {
            fprintf(f, "0\n");
            continue;
        }

        size_t const * const jv_begin = Te.jv.data()+Te.pv[i];
        size_t const * const jv_end = Te.jv.data()+Te.pv[i+1];

        double const r = calc_rate(i, model, jv_begin, jv_end);

        fprintf(f, "%f\n", r);
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

    SpMat const Te = read_data(opt.te_path);

    Model const model = read_model(opt.model_path);

    predict(Te, model, opt.output_path);
    
    return EXIT_SUCCESS;
}
