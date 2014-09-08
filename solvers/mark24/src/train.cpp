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
#include "gbdt.h"

namespace {

struct Option
{
    Option() : nr_trees(2), save_model(true) {}
    std::string Tr_path, model_path, Va_path;
    size_t nr_trees;
    bool save_model;
};

std::string train_help()
{
    return std::string(
"usage: mark24 [<options>] <train_path> [<model_path>]\n"
"\n"
"options:\n"
"-t <size>: you know\n"
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
            opt.nr_trees = std::stoi(args[++i]);
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.Va_path = args[++i];
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

    printf("reading data...");
    fflush(stdout);
    Problem const Tr = read_data(opt.Tr_path);
    Problem const Va = read_data(opt.Va_path);
    printf("done\n");
    fflush(stdout);

    GBDT gbdt(opt.nr_trees);
    gbdt.fit(Tr);

    return EXIT_SUCCESS;
}
