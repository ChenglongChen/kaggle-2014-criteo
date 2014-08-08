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

double logistic_func(double const t)
{
    return 1/(1+exp(-t));
}

double logistic_func_dt(double const t)
{
    double const expt = exp(-t);
    return expt/((1+expt)*(1+expt));
}

size_t const kSize = 100;

inline double sum1(double const t)
{
    double value = t;
    double sum = t;
    for(size_t i = 1; i <= kSize; ++i)
    {
        value = value*t*t/(2*static_cast<double>(i)+1);
        sum += value;
    }

    return sum;
}

inline double sum2(double const t)
{
    double value = 1;
    double sum = 1;
    for(size_t i = 1; i < kSize; ++i)
    {
        value = value*t*t/(2*static_cast<double>(i-1)+1);
        sum += value;
    }

    return sum;
}

double normal_dist(double const t)
{
    return 0.5+(1/(sqrt(2*3.1416)))*exp(-t*t/2)*sum1(t);
}

double normal_dist_dt(double const t)
{
    return (1/(sqrt(2*3.1416)))*(exp(-t*t/2)*(-t)*sum1(t)+exp(-t*t/2)*sum2(t));
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
    double (*calc_prob) (double const t) = &logistic_func;
    double (*calc_prob_dt) (double const t) = &logistic_func_dt;

    FILE *f = open_c_file("out.txt", "w");

    Model model(Tr.n);

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
            
            double t = 0;
            for(size_t idx = Tr.P[i]; idx < Tr.P[i+1]; ++idx)
                t += model.W[Tr.J[idx]];
            
            double const prob = calc_prob(t);

            Tr_loss -= y*log(prob) + (1-y)*log(1-prob);
               
            double const kappa = -(y*(1/prob)+(y-1)*(1/(1-prob)))*calc_prob_dt(t);

            for(size_t idx = Tr.P[i]; idx < Tr.P[i+1]; ++idx)
            {
                double &w = model.W[Tr.J[idx]];
                double &wG = model.WG[Tr.J[idx]];
                double const g = opt.lambda*w + kappa;
                wG += g*g;
                w = w - opt.eta*qrsqrt(static_cast<float>(wG))*g;
            }
        }

        printf("%3ld %7.5f", iter, Tr_loss/static_cast<double>(Tr.Y.size()));

        if(Va.Y.size() != 0)
        {
            double Va_loss = 0;
            for(size_t i = 0; i < Va.Y.size(); ++i)
            {
                double const y = static_cast<double>(Va.Y[i]);

                double t = 0;
                for(size_t idx = Va.P[i]; idx < Va.P[i+1]; ++idx)
                    t += model.W[Va.J[idx]];
                
                double const prob = calc_prob(t);

                Va_loss -= y*log(prob) + (1-y)*log(1-prob);

                if(iter == opt.iter-1)
                    fprintf(f, "%lf\n", prob);
            }
            printf(" %7.5f", Va_loss/static_cast<double>(Va.Y.size()));
        }
        printf("\n");
    }

    fclose(f);

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
