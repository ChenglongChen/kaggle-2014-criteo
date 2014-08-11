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
    Option() : lambda(0.0), eta(0.1), iter(10) {}
    std::string Tr_path, model_path, Va_path;
    double lambda, eta;
    size_t iter;
};

std::string train_help()
{
    return std::string(
"usage: sgd-poly2-train [<options>] <train_path>\n"
"\n"
"options:\n"
"-l <penalty>: you know\n"
"-t <iteration>: you know\n"
"-r <eta>: you know\n"
"-v <path>: you know\n");
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

inline double logistic_func_dt(double const t)
{
    double const expt = exp(-t);
    return expt/((1+expt)*(1+expt));
}

inline float qrsqrt(float x)
{
  float xhalf = 0.5f*x;
  uint32_t i;
  std::memcpy(&i, &x, sizeof(i));
  i = 0x5f375a86 - (i>>1);
  std::memcpy(&x, &i, sizeof(i));
  x = x*(1.5f - xhalf*x*x);
  return x;
}

Model train(SpMat const &Tr, SpMat const &Va, Option const &opt)
{
    double const lambda = opt.lambda/static_cast<double>(Tr.Y.size());

    Model model;

    std::vector<size_t> order(Tr.Y.size());
    for(size_t i = 0; i < Tr.Y.size(); ++i)
        order[i] = i;

    for(size_t iter = 0; iter < opt.iter; ++iter)
    {
        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end());
        for(size_t i_ = 0; i_ < order.size(); ++i_)
        {
            size_t const i = order[i_];

            double const y = static_cast<double>(Tr.Y[i]);
            
            double const t = wTx(Tr, model, i);

            double const prob = logistic_func(t);

            Tr_loss -= y*log(prob) + (1-y)*log(1-prob);
               
            double const kappa = -(y*(1/prob)+(y-1)*(1/(1-prob)))*logistic_func_dt(t);

            //size_t const nnz = Tr.P[i+1] - Tr.P[i];
            //double const coef = qrsqrt(static_cast<float>(nnz*(nnz+1)/2));
            for(size_t idx1 = Tr.P[i]; idx1 < Tr.P[i+1]; ++idx1)
            {
                for(size_t idx2 = idx1+1; idx2 < Tr.P[i+1]; ++idx2)
                {
                    size_t const w_idx = (Tr.J[idx1]*Tr.J[idx2])%kW_SIZE;
                    double &w = model.W[w_idx];
                    double &wG = model.WG[w_idx];
                    double const g = lambda*w + kappa*Tr.X[idx1]*Tr.X[idx2];
                    wG += g*g;
                    w = w - opt.eta*qrsqrt(static_cast<float>(wG))*g;
                }
            }
        }

        printf("%3ld %7.5f", iter, Tr_loss/static_cast<double>(Tr.Y.size()));

        if(Va.Y.size() != 0)
        {
            double Va_loss = 0;
            for(size_t i = 0; i < Va.Y.size(); ++i)
            {
                double const y = static_cast<double>(Va.Y[i]);

                double const t = wTx(Va, model, i);
                
                double const prob = logistic_func(t);

                Va_loss -= y*log(prob) + (1-y)*log(1-prob);
            }
            printf(" %7.5f", Va_loss/static_cast<double>(Va.Y.size()));
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

    SpMat const Tr = read_data(opt.Tr_path);

    SpMat Va;
    if(!opt.Va_path.empty())
        Va = read_data(opt.Va_path);

    Model model = train(Tr, Va, opt);

    save_model(model, opt.model_path);

    return EXIT_SUCCESS;
}
