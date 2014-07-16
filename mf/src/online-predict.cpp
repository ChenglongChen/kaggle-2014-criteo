#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <cmath>

#include "common.h"
#include "ftrl.h"
#include "eval.h"

std::tuple<std::vector<int>, std::vector<double>> 
predict(std::string const &te_path, 
        std::string const &out_path, 
        std::vector<double> const &W)
{
    uint const kMaxLineSize = 1000000;
    FILE *f = open_c_file(te_path.c_str(), "r");
    FILE *f_out = open_c_file(out_path.c_str(), "w");
    char line[kMaxLineSize];

    std::vector<int> y;
    std::vector<double> dec_vals;
    while(fgets(line, kMaxLineSize, f) != nullptr)
    {
        char *p = strtok(line, " \t");
        int const y1 = (atoi(p)>0)? 1 : 0;

        double dec_val1 = 0;

        while(1)
        {
            char *idx_char = strtok(nullptr,":");
            char *val_char = strtok(nullptr," \t");

            if(val_char == nullptr || *val_char == '\n')
                break;

            size_t const idx1 = atoi(idx_char);
            double const val1 = atof(val_char);

            if(idx1 > W.size())
                continue;

            dec_val1 += W[idx1-1]*(val1);
        }

        fprintf(f_out, "%lf\n", 1/(1+exp(-dec_val1)));

        y.push_back(y1);
        dec_vals.push_back(dec_val1);
    }

    fclose(f);

    return std::make_tuple(y, dec_vals);
}

double calc_density(std::vector<double> const &W)
{
    uint nnz = 0;
    for (auto w : W)
        if (w != 0)
            ++nnz;
    return (double)nnz/(double)W.size();
}

int main(const int argc, char * const * const argv) 
{
    if(argc != 3 && argc != 5)
    {
        std::cout << "usage: online-predict [-r <sample rate>] test_path output_path"
                  << std::endl;
        return EXIT_FAILURE;
    }
    double rate = 1.0;
    std::string te_path, out_path;
    if(argc == 3)
    {
        te_path = std::string(argv[1]);
        out_path = std::string(argv[2]);
    }
    else
    {
        rate = atof(argv[2]);
        te_path = std::string(argv[3]);
        out_path = std::string(argv[4]);
    }

    std::vector<int> y;
    std::vector<double> dec_vals;

    FTRL learner;
    try
    {
        learner.load();
    }
    catch(std::runtime_error const &e)
    {
        printf("[warning] model not exists");
        return EXIT_FAILURE;
    }

    std::tie(y, dec_vals) = predict(te_path, out_path, learner.get_W());

    for(auto &y1 : y)
        if(y1 == 0)
            y1 = -1;

    std::cout << "AUC: " << calc_auc(dec_vals, y) << std::endl;
    std::cout << "Acc: " << calc_accuracy(dec_vals, y) << std::endl;
    std::cout << "CTR MAE: " << calc_ctr_mae(dec_vals, y, rate) << std::endl;
    std::cout << "log loss: " << calc_log_loss(dec_vals, y, rate) << std::endl;
    std::cout << "density: " << calc_density(learner.get_W()) << std::endl;

    return EXIT_SUCCESS;
}
