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
#include "gbdt.h"

namespace {

struct Option
{
    Option() : nr_trees(10), nr_threads(1) {}
    std::string Tr_path, Va_path;
    size_t nr_trees, nr_threads;
};

std::string train_help()
{
    return std::string(
"usage: mark26 [<options>] <train_path> \n"
"\n"
"options:\n"
"-d <depth>: you know\n"
"-s <nr_threads>: you know\n"
"-t <nr_tree>: you know\n"
"-v <path>: you know\n");
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
        if(args[i].compare("-d") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            TreeNode::max_depth = std::stoi(args[++i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_trees = std::stoi(args[++i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_threads = std::stoi(args[++i]);
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.Va_path = args[++i];
        }
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw std::invalid_argument("training data not specified");

    opt.Tr_path = args[i++];

    return opt;
}

void write(
    DenseColMat const &problem, GBDT const &gbdt, std::string const &path)
{
    FILE *f = open_c_file(path, "w");

    for(size_t i = 0; i < problem.nr_instance; ++i)
    {
        std::vector<float> x(kNR_FEATURE);
        for(size_t j = 0; j < kNR_FEATURE; ++j)
            x[j] = problem.X[j][i];

        std::vector<size_t> indices = gbdt.get_indices(x.data());

        for(size_t t = 0; t < indices.size(); ++t)
            fprintf(f, "%ld:%ld ", t+1, indices[t]);
        fprintf(f, "\n");
    }

    fclose(f);
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
    DenseColMat const Tr = read_dcm(opt.Tr_path);
    DenseColMat const Va = read_dcm(opt.Va_path);
    printf("done\n");
    fflush(stdout);

	omp_set_num_threads(static_cast<int>(opt.nr_threads));

    GBDT gbdt(opt.nr_trees);
    gbdt.fit(Tr, Va);

    write(Tr, gbdt, opt.Tr_path+".gbdt");
    write(Va, gbdt, opt.Va_path+".gbdt");

    return EXIT_SUCCESS;
}
