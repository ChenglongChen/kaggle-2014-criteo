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
    Option() : eta(0.1f), iter(5) {}
    std::string Tr_p1_path, Tr_p2_path, model_path, Va_p1_path, Va_p2_path;
    float eta;
    size_t iter;
};

std::string train_help()
{
    return std::string(
"usage: sgd-poly2-train [<options>] <train_path>\n"
"\n"
"options:\n"
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
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.Va_p1_path = args[++i];
            option.Va_p2_path = args[++i];
        }
        else
        {
            break;
        }
    }

    if(i >= argc-2)
        throw std::invalid_argument("training data not specified");
    option.Tr_p1_path = args[i++];
    option.Tr_p2_path = args[i++];
    option.model_path = std::string(args[i]);

    return option;
}

Model train(SpMat const &Tr_p1, SpMat const &Tr_p2, SpMat const &Va_p1, 
    SpMat const &Va_p2, Option const &opt)
{
    Model model;

    std::vector<size_t> order(Tr_p1.Y.size());
    for(size_t i = 0; i < Tr_p1.Y.size(); ++i)
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

            float const y = Tr_p1.Y[i];
            
            float t = 0;
            
            t += wTx_p1(Tr_p1, model, i);
            t += wTx_p2(Tr_p2, model, i);

            float const expnyt = static_cast<float>(exp(-y*t));

            Tr_loss += log(1+expnyt);
               
            float const kappa = -y*expnyt/(1+expnyt);

            wTx_p1(Tr_p1, model, i, kappa, opt.eta, true);
            wTx_p2(Tr_p2, model, i, kappa, opt.eta, true);
        }

        printf("%3ld %8.2f %10.5f", iter, timer.toc(), 
            Tr_loss/static_cast<double>(Tr_p1.Y.size()));

        if(Va_p1.Y.size() != 0)
            printf(" %10.5f", predict(Va_p1, Va_p2, model));

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

    SpMat const Tr_p1 = read_data(opt.Tr_p1_path);
    SpMat const Tr_p2 = read_data(opt.Tr_p2_path);

    SpMat Va_p1, Va_p2;
    if(!opt.Va_p1_path.empty() && !opt.Va_p2_path.empty())
    {
        Va_p1 = read_data(opt.Va_p1_path);
        Va_p2 = read_data(opt.Va_p2_path);
    }

    Model model = train(Tr_p1, Tr_p2, Va_p1, Va_p2, opt);

    save_model(model, opt.model_path);

    return EXIT_SUCCESS;
}
