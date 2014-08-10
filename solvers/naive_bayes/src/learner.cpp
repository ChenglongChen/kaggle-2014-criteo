#include <fstream>

#include "learner.h"

void Learner::convert(std::string const &path)
{
    std::ofstream f(path);

    //f << "solver_type L1R_LR\n";
    //f << "nr_class 2\n";
    //f << "label 1 -1\n";
    //f << "nr_feature " << W.size() << "\n";
    //f << "bias -1\n";
    //f << "w\n";

    f << W.size() << "\n";

    for(auto w : W)
        f << w << "\n";
}

std::vector<double> const &Learner::get_W()
{
    return W;
}

std::string const Learner::model_path = ".naive_beyes.bin";
