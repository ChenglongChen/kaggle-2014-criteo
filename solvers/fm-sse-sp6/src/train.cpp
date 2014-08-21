#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

#include "common.h"
#include "timer.h"

namespace {

struct Option
{
    Option() : eta(0.1f), lambda(0.00001f), iter(5), k(4), k_real(4), nr_threads(1), ignore(-1), save_model(true) {}
    std::string Tr_path, model_path, Va_path;
    float eta, lambda;
    size_t iter, k, k_real, nr_threads, ignore;
    bool save_model;
};

std::string train_help()
{
    return std::string(
"usage: sgd-poly2-train [<options>] <train_path>\n"
"\n"
"options:\n"
"-l <labmda>: you know\n"
"-k <dimension>: you know\n"
"-t <iteration>: you know\n"
"-r <eta>: you know\n"
"-s <nr_threads>: you know\n"
"-v <path>: you know\n"
"-q: you know\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    size_t const argc = args.size();

    if(argc == 0)
        throw std::invalid_argument(train_help());

    Option opt; 

    size_t i = 0;
    for(; i < argc; ++i)
    {
        if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.iter = std::stoi(args[++i]);
        }
        else if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.k_real = std::stoi(args[++i]);
            opt.k = static_cast<size_t>(ceil(static_cast<float>(opt.k_real)/4.0f))*4;
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.eta = std::stof(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda = std::stof(args[++i]);
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.Va_path = args[++i];
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_threads = std::stoi(args[++i]);
        }
        else if(args[i].compare("-i") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.ignore = std::stoi(args[++i])-1;
        }
        else if(args[i].compare("-q") == 0)
        {
            opt.save_model = false;
        }
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw std::invalid_argument("training data not specified");

    opt.Tr_path = args[i++];

    if(i < argc)
    {
        opt.model_path = std::string(args[i]);
    }
    else if(i == argc)
    {
        const char *ptr = strrchr(&*opt.Tr_path.begin(),'/');
        if(!ptr)
            ptr = opt.Tr_path.c_str();
        else
            ++ptr;
        opt.model_path = std::string(ptr) + ".model";
    }
    else
    {
        throw std::invalid_argument("invalid argument");
    }

    return opt;
}

void init_model(Model &model, size_t const k_real)
{
    size_t const k = model.k;
    float const coef = 
        static_cast<float>(0.5/sqrt(static_cast<double>(k_real)));

    float * w = model.W.data();
    for(size_t j = 0; j < model.n; ++j)
    {
        for(size_t f = 0; f < kF_SIZE; ++f)
        {
            for(size_t d = 0; d < k_real; ++d, ++w)
                *w = coef*static_cast<float>(drand48());
            for(size_t d = k_real; d < k; ++d, ++w)
                *w = 0;
            for(size_t d = k; d < 2*k; ++d, ++w)
                *w = 1;
        }
    }
}

void train(SpMat const &Tr, SpMat const &Va, Model &model, Option const &opt)
{
    std::vector<size_t> order(Tr.Y.size());
    for(size_t i = 0; i < Tr.Y.size(); ++i)
        order[i] = i;

    Timer timer;
    for(size_t iter = 0; iter < opt.iter; ++iter)
    {
        timer.tic();

        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end());
#pragma omp parallel for schedule(static)
        for(size_t i_ = 0; i_ < order.size(); ++i_)
        {
            size_t const i = order[i_];

            float const y = Tr.Y[i];
            
            float const t = wTx(Tr, model, i, opt.ignore);

            float const expnyt = static_cast<float>(exp(-y*t));

            Tr_loss += log(1+expnyt);
               
            float const kappa = -y*expnyt/(1+expnyt);

            wTx(Tr, model, i, opt.ignore, kappa, opt.eta, opt.lambda, true);
        }

        printf("%3ld %8.2f %10.5f", iter, timer.toc(), 
            Tr_loss/static_cast<double>(Tr.Y.size()));

        if(Va.Y.size() != 0)
            printf(" %10.5f", predict(Va, model, opt.ignore));

        printf("\n");
        fflush(stdout);
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

    printf("ignore: %ld\n", opt.ignore);

    printf("reading data...");
    fflush(stdout);
    SpMat const Tr = read_data(opt.Tr_path);
    SpMat const Va = read_data(opt.Va_path);
    printf("done\n");
    fflush(stdout);

    printf("initializing model...");
    fflush(stdout);
    Model model(Tr.n, opt.k);

    init_model(model, opt.k_real);
    printf("done\n");
    fflush(stdout);

	omp_set_num_threads(static_cast<int>(opt.nr_threads));

    train(Tr, Va, model, opt);

    if(opt.save_model)
        save_model(model, opt.model_path);

    return EXIT_SUCCESS;
}
