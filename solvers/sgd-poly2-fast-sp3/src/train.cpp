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
    Option() : eta(0.1f), iter(5), threshold(1) {}
    std::string Tr_path, model_path, Va_path;
    float eta;
    size_t iter;
    int threshold;
};

std::string train_help()
{
    return std::string(
"usage: sgd-poly2-train [<options>] <train_path>\n"
"\n"
"options:\n"
"-t <iteration>: you know\n"
"-h <threshold>: you know\n"
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
        if(args[i].compare("-t") == 0)
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
        else if(args[i].compare("-h") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.threshold = std::stoi(args[++i]);
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

Model train(SpMat const &Tr, SpMat const &Va, Option const &opt)
{
    Model model;

    std::vector<size_t> order(Tr.Y.size());
    for(size_t i = 0; i < Tr.Y.size(); ++i)
        order[i] = i;

    for(size_t i = 0; i < Tr.Y.size(); ++i)
    {
        for(size_t idx1 = Tr.P[i]; idx1 < Tr.P[i+1]; ++idx1)
        {
            size_t const j1 = Tr.JX[idx1].j;
            for(size_t idx2 = idx1+1; idx2 < Tr.P[i+1]; ++idx2)
            {
                size_t const j2 = Tr.JX[idx2].j;
                size_t const w_idx = calc_w_idx(j1,j2);
                ++model.W[w_idx].mask;
            }
        }
    }

    for(auto &w1 : model.W)
        w1.mask = w1.mask >= opt.threshold? 1 : 0;

    Timer timer;
    for(size_t iter = 0; iter < opt.iter; ++iter)
    {
        timer.tic();

        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end());
        for(size_t i_ = 0; i_ < order.size(); ++i_)
        {
            size_t const i = order[i_];

            float const y = Tr.Y[i];
            
            float const t = wTx(Tr, model, i);

            float const expnyt = static_cast<float>(exp(-y*t));

            Tr_loss += log(1+expnyt);
               
            float const kappa = -y*expnyt/(1+expnyt);

            wTx(Tr, model, i, kappa, opt.eta, true);
        }

        printf("%3ld %8.2f %10.5f", iter, timer.toc(), 
            Tr_loss/static_cast<double>(Tr.Y.size()));

        if(Va.Y.size() != 0)
            printf(" %10.5f", predict(Va, model));

        printf("\n");
        fflush(stdout);
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