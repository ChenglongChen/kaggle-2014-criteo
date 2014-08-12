#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "common.h"
#include "timer.h"

namespace {

struct Option
{
    Option() : lambda(0.0f), eta(0.1f), iter(10) {}
    std::string Tr_path, model_path, Va_path;
    float lambda, eta;
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

inline float logistic_func_dt(float const t)
{
    float const expt = static_cast<float>(exp(-t));
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
    float const lambda = 
        static_cast<float>(opt.lambda/static_cast<double>(Tr.Y.size()));

    Model model;

    std::vector<size_t> order(Tr.Y.size());
    for(size_t i = 0; i < Tr.Y.size(); ++i)
        order[i] = i;

    Timer timer;
    for(size_t iter = 0; iter < opt.iter; ++iter)
    {
        timer.tic();

        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end());
        for(size_t i_ = 0; i_ < order.size(); ++i_)
        {
            size_t const i = order[i_];

            float const y = static_cast<float>(Tr.Y[i]);
            
            float const t = wTx(Tr, model, i);

            float const prob = logistic_func(t);

            Tr_loss -= y*log(prob) + (1-y)*log(1-prob);
               
            float const kappa = -(y*(1/prob)+(y-1)*(1/(1-prob)))*logistic_func_dt(t);

            for(size_t idx1 = Tr.P[i]; idx1 < Tr.P[i+1]; ++idx1)
            {
                for(size_t idx2 = idx1+1; idx2 < Tr.P[i+1]; ++idx2)
                {
                    size_t const w_idx = cantor(Tr.J[idx1],Tr.J[idx2])%kW_SIZE;
                    float &w = model.W[w_idx];
                    float &wG = model.WG[w_idx];
                    float const g = lambda*w + kappa*Tr.X[idx1]*Tr.X[idx2];
                    wG += g*g;
                    w = w - opt.eta*qrsqrt(wG)*g;
                }
            }
        }

        printf("%3ld %7.2f %10.5f", iter, timer.toc(), Tr_loss/static_cast<double>(Tr.Y.size()));

        if(Va.Y.size() != 0)
        {
            double Va_loss = 0;
            for(size_t i = 0; i < Va.Y.size(); ++i)
            {
                float const y = static_cast<float>(Va.Y[i]);

                float const t = wTx(Va, model, i);
                
                float const prob = logistic_func(t);

                Va_loss -= y*log(prob) + (1-y)*log(1-prob);
            }
            printf(" %10.5f", Va_loss/static_cast<double>(Va.Y.size()));
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
