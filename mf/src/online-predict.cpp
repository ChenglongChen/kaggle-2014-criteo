#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include "ftrl.h"
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
predict(SpMat const &spmat, Model const &model, std::string const &output_path)
{
    size_t const k = model.k;
    
    FILE *f = open_c_file(output_path, "w");     

    float const * const P = model.P.data();
    for(auto p = spmat.pv.begin(); p < spmat.pv.end()-1; ++p)
    {
        size_t nnz = *(p+1)-(*p);
        if(nnz <= 1)
        {
            fprintf(f, "0\n");
            continue;
        }

        size_t const * const jv_begin = spmat.jv.data()+(*p);
        size_t const * const jv_end = spmat.jv.data()+(*(p+1));
        
        double r = 0;
        for(size_t const *u = jv_begin; u != jv_end; ++u)
        {
            float const * const pu = P+(*u)*k;
            for(size_t const *v = u; v != jv_end; ++v) 
            {
                float const * const pv = P+(*v)*k;
                for(size_t d = 0; d < k; ++d)
                    r += (*(pu+d))*(*(pv+d));
            }
        }

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
