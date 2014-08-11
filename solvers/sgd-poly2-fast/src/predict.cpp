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

void predict(SpMat const &Te, Model const &model, std::string const &output_path)
{
    FILE *f = open_c_file(output_path, "w");
    double Te_loss = 0;
    for(size_t i = 0; i < Te.Y.size(); ++i)
    {
        double const y = static_cast<double>(Te.Y[i]);

        double const t = wTx(Te, model, i);
        
        double const prob = logistic_func(t);

        Te_loss -= y*log(prob) + (1-y)*log(1-prob);

        fprintf(f, "%lf\n", prob);
    }
    printf("logloss = %7.5f\n", Te_loss/static_cast<double>(Te.Y.size()));
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

    SpMat const Te = read_data(opt.Te_path);

    Model model = load_model(opt.model_path);
    
    predict(Te, model, opt.output_path);

    return EXIT_SUCCESS;
}
