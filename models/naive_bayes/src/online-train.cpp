#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ftrl.h"
#include "common.h"

struct Option 
{
    Option() : alpha(0.1), beta(1), lambda1(1), lambda2(0), force(false) {} ;
    std::string tr_path, model_path;
    double alpha, beta, lambda1, lambda2;
    bool force;
};

std::string train_help()
{
    return std::string(
"usage: online-train [<options>] <train_path>\n"
"\n"
"options:\n"
"-a <alpha>: set alpha\n"
"-b <beta>: set beta\n"
"-l <lambda1>: set lambda for L1 regularization\n"
"-c <lambda2>: set lambda for L2 regularization\n"
"-f: discard an existing model\n"
"\n"
"The model is stored in \".online_model\".\n"
"Use \"online-extract\" to convert it to a LIBLINEAR compatible model.\n");
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
        if(args[i].compare("-a") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.alpha = std::stof(args[++i]);
        }
        else if(args[i].compare("-b") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.beta = std::stof(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.lambda1 = std::stof(args[++i]);
        }
        else if(args[i].compare("-c") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            option.lambda2 = std::stof(args[++i]);
        }
        else if(args[i].compare("-f") == 0)
        {
            option.force = true;
        }
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw std::invalid_argument("training data not specified");

    option.tr_path = args[i];

    return option;
}

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}

void learn(FTRL * const learner, std::string const tr_path)
{
    uint const kMaxLineSize = 1000000;
	FILE *f = open_c_file(tr_path.c_str(), "r");
    char line[kMaxLineSize];

    std::vector<uint> idx;
    std::vector<double> val;
	while(fgets(line, kMaxLineSize, f) != nullptr)
	{
		char *p = strtok(line, " \t");
        int const y = (atoi(p)>0)? 1 : 0;

        idx.clear();
        val.clear();
		while(1)
		{
			char *idx1 = strtok(nullptr,":");
			char *val1 = strtok(nullptr," \t");

			if(val1 == nullptr || *val1 == '\n')
				break;

            idx.push_back(static_cast<uint>(atoi(idx1))-1);
            val.push_back(static_cast<double>(atof(val1)));
		}
        learner->update(y, idx, val);
	}

    fclose(f);
}

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

    FTRL learner;

    if(!opt.force)
    {
        try
        {
            learner.load();
        }
        catch(std::runtime_error const &e)
        {
            printf("[warning] no previous model exists."
                "using a brand new model\n");
        }
    }
    else
    {
        printf("[warning] using a brand new model\n");
    }

    learn(&learner, opt.tr_path);

    learner.save();

    return EXIT_SUCCESS;
}
