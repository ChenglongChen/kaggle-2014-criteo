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
    Option() : lambda(1.0), eta(0.01), iter(10) {}
    std::string Tr_path, model_path, Va_path;
    double lambda, eta;
    size_t iter;
};

std::string train_help()
{
    return std::string(
"usage: online-train [<options>] <train_path>\n"
"\n"
"options:\n"
"-l <penalty>: you know\n"
"-t <iteration>: you know\n"
"-r <eta>: you know\n"
"-v <path>: you know\n"
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
        if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.lambda = std::stof(args[++i]);
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
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw std::invalid_argument("training data not specified");

    option.Tr_path = args[i++];

    if(i < argc)
    {
        option.model_path = std::string(args[i]);
    }
    else if(i == argc)
    {
        const char *ptr = strrchr(&*option.Tr_path.begin(),'/');
        if(!ptr)
            ptr = option.Tr_path.c_str();
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

inline double calc_prob(double const t)
{
    return 1/(1+exp(-t));
}

inline double calc_prob_dt(double const t)
{
    double const expt = exp(-t);
    return expt/((1+expt)*(1+expt));
}

Model train(SpMat const &Tr, SpMat const &Va, Option const &opt)
{
    Model model(Tr.n);

    std::vector<size_t> order(Tr.Y.size());
    for(size_t i = 0; i < Tr.Y.size(); ++i)
        order[i] = i;

    for(size_t t = 0; t < opt.iter; ++t)
    {
        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end());
        for(size_t i_ = 0; i_ < order.size(); ++i_)
        {
            size_t const i = order[i_];

            int const y = Tr.Y[i];
            
            double t = 0;
            for(size_t idx = Tr.P[i]; idx < Tr.P[i+1]; ++idx)
                t += model.W[Tr.J[idx]];
            
            double const prob = calc_prob(t);

            //printf("DB1: %lf, %lf\n", t, prob);

            Tr_loss -= y*log(prob) + (1-y)*log(1-prob);
               
            double const kappa = -(y*(1/prob)+(y-1)*(1/(1-prob)))*calc_prob_dt(t);

            for(size_t idx = Tr.P[i]; idx < Tr.P[i+1]; ++idx)
            {
                double &w = model.W[Tr.J[idx]];
                double const g = opt.lambda*w + kappa;
                w = w - opt.eta*g;
            }
        }

        printf("%3ld %7.5f", t, Tr_loss/static_cast<double>(Tr.Y.size()));

        //if(Va.Y.size() != 0)
        //{
        //    double Va_loss = 0;
        //    for(size_t i = 0; i < Va.Y.size(); ++i)
        //    {
        //        size_t const * const x = Va.X.data()+i*FieldSizes.size();
        //        float const y = static_cast<float>(Va.Y[i]);

        //        float const r = calc_rate(model, x);

        //        float const expyr = static_cast<float>(ALPHA*exp(-BETA*exp(-GAMMA*r)));

        //        Va_loss -= (y==1)? log(1-expyr) : log(expyr);
        //    }
        //    printf(" %7.5f", Va_loss/static_cast<double>(Va.Y.size()));
        //}
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

    SpMat const Tr = read_data(opt.Tr_path);

    SpMat Va;
    if(!opt.Va_path.empty())
        Va = read_data(opt.Va_path);

    Model model = train(Tr, Va, opt);

    return EXIT_SUCCESS;
}
