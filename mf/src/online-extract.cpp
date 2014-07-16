#include <stdexcept>

#include "ftrl.h"
#include "common.h"

int main(int const argc, char const * const * const argv)
{
    if(argc != 2)
    {
        printf("usage: online-extract LIBLINEAR_model_path\n");
        return EXIT_FAILURE;
    }

    std::string liblinear_model_path(argv[1]);
        
    FTRL learner; 

    try
    {
        learner.load();
    }
    catch(std::runtime_error const &e)
    {
        printf("[error] the model does not exist\n");
        return EXIT_FAILURE;
    }

    learner.convert(liblinear_model_path);

    return EXIT_SUCCESS;
}
