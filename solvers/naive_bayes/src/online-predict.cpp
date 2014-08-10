#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <cmath>

#include "common.h"
#include "ftrl.h"
#include "eval.h"

void
predict(std::string const &te_path, 
        std::string const &out_path, 
        FTRL const &learner, 
        int const threshold)
{
    uint const kMaxLineSize = 1000000;
    FILE *f = open_c_file(te_path.c_str(), "r");
    FILE *f_out = open_c_file(out_path.c_str(), "w");
    char line[kMaxLineSize];

    double const prior = static_cast<double>(learner.pos) / 
                         static_cast<double>(learner.pos+learner.neg);

    while(fgets(line, kMaxLineSize, f) != nullptr)
    {
        strtok(line, " \t");

        double prob = prior;

        while(1)
        {
            char *idx_char = strtok(nullptr,":");
            char *val_char = strtok(nullptr," \t");

            if(val_char == nullptr || *val_char == '\n')
                break;

            size_t const idx1 = atoi(idx_char)-1;

            if(idx1 >= learner.likelihood.size())
                continue;

            double const likelihood = 
                static_cast<double>(learner.likelihood[idx1]) /
                static_cast<double>(learner.pos);

            double const evidence = 
                static_cast<double>(learner.evidence[idx1]) /
                static_cast<double>(learner.pos+learner.neg);

            if(likelihood != 0 && learner.evidence[idx1] > threshold)
                prob *= likelihood / evidence;
            //std::cout << likelihood << ", " << evidence << ", " << prob << "\n";
        }
        //std::cout << "---------------\n";
        fprintf(f_out, "%lf\n", prob);
    }

    fclose(f);
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
        std::cout << "usage: online-predict [-r threshold] test_path output_path"
                  << std::endl;
        return EXIT_FAILURE;
    }
    std::string te_path, out_path;

    int threshold = 0;
    if(argc == 3)
    {
        te_path = std::string(argv[1]);
        out_path = std::string(argv[2]);
    }
    else
    {
        threshold = atoi(argv[2]);
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

    predict(te_path, out_path, learner, threshold);

    return EXIT_SUCCESS;
}
