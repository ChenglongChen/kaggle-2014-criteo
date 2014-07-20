#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "common.h"

namespace {

struct Option 
{
    Option() : c(1.0f), eta(0.01f), k(2), iter(10), n_bar(39) {} ;
    std::string tr_path, model_path, Va_path;
    float c, eta;
    int k, iter, n_bar;
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
"-v <path>: you know\n"
"-n <n_bar>: you know\n"
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
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.Va_path = args[++i];
        }
        else if(args[i].compare("-n") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.n_bar = std::stoi(args[++i]);
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

Model train(SpMat const &Tr, SpMat const &Va, Option const &opt)
{
    size_t const k = opt.k;
    size_t const n = Tr.n;

    Model model(n, k, opt.n_bar);
    float * const P = model.P.data();
    float * const Q = model.Q.data();

    for(auto &p : model.P)
        p = 0.01f*static_cast<float>(drand48());
    for(auto &q : model.Q)
        q = 0.01f*static_cast<float>(drand48());

    std::vector<size_t> order(Tr.pv.size()-1);
    for(size_t i = 0; i < Tr.pv.size()-1; ++i)
        order[i] = i;

    std::vector<float> sum(k, 0);
    for(int t = 0; t < opt.iter; ++t)
    {
        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end());
        for(size_t x = 0; x < Tr.pv.size()-1; ++x)
        {
            size_t const i = order[x];//rand()%(Tr.pv.size()-1);

            auto y = Tr.yv.begin()+i;
            size_t nnz = Tr.pv[i+1]-Tr.pv[i];
            if(nnz <= 1)
            {
                Tr_loss += log(2);
                continue;
            }

            size_t const * const jv_begin = Tr.jv.data()+Tr.pv[i];
            size_t const * const jv_end = Tr.jv.data()+Tr.pv[i+1];
            
            float const r = calc_rate(i, model, jv_begin, jv_end);
            float const expyr = 
                static_cast<float>(exp(-static_cast<float>(*y)*r));

            float const alpha = -static_cast<float>(*y)*expyr/(1+expyr);

            Tr_loss += log(1+expyr);

            size_t cell_idx = 0;
            for(size_t const *u = jv_begin; u != jv_end; ++u)
            {
                for(size_t const *v = u+1; v != jv_end; ++v, ++cell_idx) 
                {
                    size_t const offset = cell_idx*n*k;
                    float * const pu = P+offset+(*u)*k;
                    float * const qv = Q+offset+(*v)*k;
                    for(size_t d = 0; d < k; ++d)
                    {
                        float const tmp = pu[d];
                        pu[d] = pu[d] - opt.eta*(alpha*qv[d]+static_cast<float>(opt.c*pu[d]));
                        qv[d] = qv[d] - opt.eta*(alpha*tmp  +static_cast<float>(opt.c*qv[d]));
                    }
                }
            }
        }

        printf("%3d %7.5f", t, Tr_loss/static_cast<double>(Tr.pv.size()-1));

        if(Va.n != 0)
        {
            double Va_loss = 0;
            auto y = Va.yv.begin();
            for(size_t i = 0; i < Va.pv.size()-1; ++i, ++y)
            {
                size_t nnz = Va.pv[i+1]-Va.pv[i];
                if(nnz <= 1)
                {
                    Va_loss += log(2);
                    continue;
                }

                size_t const * const jv_begin = Va.jv.data()+Va.pv[i];
                size_t const * const jv_end = Va.jv.data()+Va.pv[i+1];
            
                double const r = calc_rate(i, model, jv_begin, jv_end);

                Va_loss += log(1+exp(-(*y)*r));
            }
            printf(" %7.5f", Va_loss/static_cast<double>(Va.pv.size()-1));
        }
        printf("\n");
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

    SpMat Va;
    if(!opt.Va_path.empty())
        Va = read_data(opt.Va_path);

    Model model = train(Tr, Va, opt);

    save_model(model, opt.model_path);

    return EXIT_SUCCESS;
}
