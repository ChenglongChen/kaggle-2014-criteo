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

namespace {

struct Option
{
    std::string Te_path, model_path, output_path;
};

std::string train_help()
{
    return std::string(
"usage: sgd-poly2-predict <test_path> <model_path> <output_path>\n"
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

    if(argc != 3)
        throw std::invalid_argument(train_help());

    Option option; 
    option.Te_path = args[0];
    option.model_path = args[1];
    option.output_path = args[2];

    return option;
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

	omp_set_num_threads(1);

    SpMat const Te = read_data(opt.Te_path);

    Model model = load_model(opt.model_path);
    
    double const Te_loss = predict(Te, model, opt.output_path);

    printf("logloss = %f\n", Te_loss);

    return EXIT_SUCCESS;
}
