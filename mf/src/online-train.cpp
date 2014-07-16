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
    Option() : c(1.0f), eta(0.01f), k(2), iter(10) {} ;
    std::string tr_path, model_path;
    float c, eta;
    int k, iter;
};

std::string train_help()
{
    return std::string(
"usage: online-train [<options>] <train_path>\n"
"\n"
"options:\n"
"-k <dim>: you know\n"
"-c <penalty>: you know\n"
"-t <iteration>: you know\n"
"-r <eta>: you know\n"
"\n"
"Warning: current I supports only binary features\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    size_t const argc = args.size();

    if(argc == 0)
        throw std::invalid_argument(train_help());

    Option option; 

    size_t i = 0;
    for(; i < argc; ++i)
    {
        if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.k = std::stoi(args[++i]);
        }
        else if(args[i].compare("-c") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.c = std::stof(args[++i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.iter = std::stoi(args[++i]);
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.eta = std::stof(args[++i]);
        }
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw std::invalid_argument("training data not specified");

    option.tr_path = args[i++];

    if(i < argc)
    {
        option.model_path = std::string(args[i]);
    }
    else if(i == argc)
    {
        const char *ptr = strrchr(&*option.tr_path.begin(),'/');
        if(!ptr)
            ptr = option.tr_path.c_str();
        else
            ++ptr;
        option.model_path = std::string(ptr) + ".model";
    }
    else
    {
        throw std::invalid_argument("invalid argument");
    }

    return option;
}

Model train(SpMat const &spmat, Option const &opt)
{
    size_t const k = opt.k;

    Model model(spmat.n, k);
    float * const P = model.P.data();
    for(size_t i = 0; i < k*spmat.n; ++i)
        P[i] = 0.1f*static_cast<float>(drand48());


    std::vector<float> sum(k, 0);
    for(int t = 0; t < opt.iter; ++t)
    {
        auto y = spmat.yv.begin();
        for(auto p = spmat.pv.begin(); p < spmat.pv.end()-1; ++p, ++y)
        {
            size_t nnz = *(p+1)-(*p);
            if(nnz <= 1)
                continue;

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

            float const alpha 
                = static_cast<float>(-(*y)*exp(-(*y)*r)/(1+exp(-(*y)*r)));

            sum.assign(k, 0);
            for(size_t const *u = jv_begin; u != jv_end; ++u)
            {
                float const * const pu = P+(*u)*k;
                for(size_t d = 0; d < k; ++d)
                    sum[d] += pu[d];
            }

            for(size_t const *u = jv_begin; u != jv_end; ++u)
            {
                float * const pu = P+(*u)*k;
                for(size_t d = 0; d < k; ++d)
                    pu[d] = pu[d] - opt.eta*(alpha*(sum[d]-pu[d])+static_cast<float>(nnz)*opt.c*pu[d]);
            }
        }
    }

    return model;
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

    SpMat Tr = read_data(opt.tr_path);

    Model model = train(Tr, opt);

    save_model(model, opt.model_path);

    return EXIT_SUCCESS;
}
