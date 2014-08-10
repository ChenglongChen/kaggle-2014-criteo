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
    Option() : lambda(1.0f), eta(0.01f), k(2), iter(10) {}
    std::string tr_path, model_path, meta_path, Va_path;
    float lambda, eta;
    size_t k, iter;
};

std::string train_help()
{
    return std::string(
"usage: online-train [<options>] <train_path>\n"
"\n"
"options:\n"
"-k <dim>: you know\n"
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
        if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.k = std::stoi(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
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

void rand_init_model(Model &model)
{
    size_t cell = 0;
    for(size_t u = 0; u < FieldSizes.size(); ++u)
    {
        for(size_t v = u+1; v < FieldSizes.size(); ++v, ++cell)
        {
            for(auto &p : model.P[cell])
                p = 0.01f*static_cast<float>(drand48());
            for(auto &q : model.Q[cell])
                q = 0.01f*static_cast<float>(drand48());
        }
    }
}

float qrsqrt(float x)
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
    Model model(opt.k, FieldSizes);
    rand_init_model(model);

    std::vector<size_t> order(Tr.Y.size());
    for(size_t i = 0; i < Tr.Y.size(); ++i)
        order[i] = i;

    for(size_t t = 0; t < opt.iter; ++t)
    {
        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end());
        for(size_t i_ = 0; i_ < Tr.Y.size(); ++i_)
        {
            size_t const i = order[i_];
            size_t const * const x = Tr.X.data()+i*FieldSizes.size();
            float const y = static_cast<float>(Tr.Y[i]);

            float const r = calc_rate(model, x);
            float const expyr = static_cast<float>(ALPHA*exp(-BETA*exp(-GAMMA*r)));
            float const kappa = static_cast<float>(ALPHA*BETA*GAMMA*y*exp(-BETA*exp(-GAMMA*y*r)-GAMMA*y*r));

            Tr_loss -= (y==1)? log(1-expyr) : log(expyr);

            for(auto f : A)
            {
                float * const w = &model.W[f][x[f]];
                float * const wG = &model.WG[f][x[f]];
                float const g = kappa+opt.lambda*(*w);
                *wG += g*g;
                float const eta = opt.eta*qrsqrt(*wG);
                *w -= eta*g;
            }

            size_t cell = 0;
            for(size_t u = 0; u < B.size(); ++u)
            {
                for(size_t v = u+1; v < B.size(); ++v, ++cell)
                {
                    float * const p = &model.P[cell][x[B[u]]*model.k];
                    float * const pG = &model.PG[cell][x[B[u]]*model.k];
                    float * const q = &model.Q[cell][x[B[v]]*model.k];
                    float * const qG = &model.QG[cell][x[B[v]]*model.k];
                    for(size_t d = 0; d < model.k; ++d)
                    {
                        float const pg = kappa*(*(q+d))+opt.lambda*(*(p+d));
                        float const qg = kappa*(*(p+d))+opt.lambda*(*(q+d));

                        *(pG+d) += pg*pg;
                        *(qG+d) += qg*qg;

                        float const eta_p = opt.eta*qrsqrt(*(pG+d));
                        float const eta_q = opt.eta*qrsqrt(*(qG+d));

                        *(p+d) -= eta_p*pg;
                        *(q+d) -= eta_q*qg;
                    }
                }
            }
        }

        printf("%3ld %7.5f", t, Tr_loss/static_cast<double>(Tr.Y.size()));

        if(Va.Y.size() != 0)
        {
            double Va_loss = 0;
            for(size_t i = 0; i < Va.Y.size(); ++i)
            {
                size_t const * const x = Va.X.data()+i*FieldSizes.size();
                float const y = static_cast<float>(Va.Y[i]);

                float const r = calc_rate(model, x);

                float const expyr = static_cast<float>(ALPHA*exp(-BETA*exp(-GAMMA*r)));

                Va_loss -= (y==1)? log(1-expyr) : log(expyr);
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

    SpMat const Tr = read_data(opt.tr_path);

    SpMat Va;
    if(!opt.Va_path.empty())
        Va = read_data(opt.Va_path);

    Model model = train(Tr, Va, opt);

    //save_model(model, opt.model_path);

    return EXIT_SUCCESS;
}