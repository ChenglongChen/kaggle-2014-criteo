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
#include "tron.h"

namespace {

struct Option
{
    Option() : lambda(1.0f), eps(0.001f), iter(15), nr_factor(4), nr_factor_real(4), nr_threads(1), save_model(true) {}
    std::string Tr_path, model_path, Va_path;
    double lambda, eps;
    size_t iter, nr_factor, nr_factor_real, nr_threads;
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
"-s <nr_threads>: you know\n"
"-e <eps>: you know\n"
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
            opt.nr_factor_real = std::stoi(args[++i]);
            //opt.nr_factor = static_cast<size_t>(ceil(static_cast<double>(opt.nr_factor_real)/4.0f))*4;
            opt.nr_factor = opt.nr_factor_real;
        }
        else if(args[i].compare("-e") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.eps = std::stof(args[++i]);
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

void init_model(Model &model, size_t const nr_factor_real)
{
    size_t const nr_factor = model.nr_factor;
    double const coef = 
        static_cast<double>(0.5/sqrt(static_cast<double>(nr_factor_real)));

    double * w = model.W.data();
    for(size_t j = 0; j < model.nr_feature; ++j)
    {
        for(size_t f = 0; f < kNR_FIELD; ++f)
        {
            for(size_t d = 0; d < nr_factor_real; ++d, ++w)
                *w = coef*static_cast<double>(drand48());
            for(size_t d = nr_factor_real; d < nr_factor; ++d, ++w)
                *w = 0;
        }
    }
}

class FactorFunc : public function
{
public:
    FactorFunc(SpMat const &spmat, Model const &model, double const lambda) 
        : spmat(spmat), model(model), lambda(lambda), 
          Z(spmat.nr_instance), D(spmat.nr_instance) {}
	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable();

private:
	void Xr(double *R, double *Xr);
	void XTr(double *R, double *XTr);

    SpMat const &spmat;
    Model const &model;
    double const lambda;
    std::vector<double> Z, D;
};

double FactorFunc::fun(double *w)
{
	Xr(w, Z.data());

	double f=0;
    for(size_t j = 0; j < static_cast<size_t>(get_nr_variable()); ++j)
		f += w[j]*w[j];
	f = lambda*f/2.0;

	for(size_t i = 0; i < spmat.nr_instance; ++i)
        f += log(1+exp(-spmat.Y[i]*Z[i]));

	return f;
}

void FactorFunc::grad(double *w, double *g)
{
	for(size_t i = 0; i < spmat.nr_instance; ++i)
	{
		Z[i] = 1/(1+exp(-spmat.Y[i]*Z[i]));
		D[i] = Z[i]*(1-Z[i]);
		Z[i] = (Z[i]-1)*spmat.Y[i];
	}
	XTr(Z.data(), g);

    for(size_t j = 0; j < static_cast<size_t>(get_nr_variable()); ++j)
		g[j] = lambda*w[j] + g[j];
}

int FactorFunc::get_nr_variable()
{
	return static_cast<int>(spmat.nr_feature*kNR_FIELD*model.nr_factor);
}

void FactorFunc::Hv(double *s, double *Hs)
{
    std::vector<double> wa(spmat.nr_instance);

	Xr(s, wa.data());
	for(size_t i = 0; i < spmat.nr_instance; ++i)
		wa[i] = D[i]*wa[i];

	XTr(wa.data(), Hs);
    for(size_t j = 0; j < static_cast<size_t>(get_nr_variable()); ++j)
		Hs[j] = lambda*s[j]+Hs[j];
}

void FactorFunc::Xr(double *R, double *Xr)
{
    size_t const nr_factor = model.nr_factor;
    double const * const W = model.W.data();

    for(size_t i = 0; i < spmat.nr_instance; ++i)
    {
        Xr[i] = 0;
        for(size_t idx1 = spmat.P[i]; idx1 < spmat.P[i+1]; ++idx1)
        {
            size_t const j1 = spmat.X[idx1].j;
            size_t const f1 = spmat.X[idx1].f;
            double const v1 = spmat.X[idx1].v;

            for(size_t idx2 = idx1+1; idx2 < spmat.P[i+1]; ++idx2)
            {
                size_t const j2 = spmat.X[idx2].j;
                size_t const f2 = spmat.X[idx2].f;
                double const v2 = spmat.X[idx2].v;

                double const * r1 = 
                    R+j1*kNR_FIELD*nr_factor+f2*nr_factor;
                double const * w2 = 
                    W+j2*kNR_FIELD*nr_factor+f1*nr_factor;

                for(size_t d = 0; d < nr_factor; ++d, ++r1, ++w2)
                    Xr[i] += (*r1)*(*w2)*v1*v2;
            }
        }
    }
}

void FactorFunc::XTr(double *R, double *XTr)
{
    size_t const nr_factor = model.nr_factor;
    double const * const W = model.W.data();

    for(size_t j = 0; j < static_cast<size_t>(get_nr_variable()); ++j)
		XTr[j] = 0;

    for(size_t i = 0; i < spmat.nr_instance; ++i)
    {
        double const r1 = R[i];
        for(size_t idx1 = spmat.P[i]; idx1 < spmat.P[i+1]; ++idx1)
        {
            size_t const j1 = spmat.X[idx1].j;
            size_t const f1 = spmat.X[idx1].f;
            double const v1 = spmat.X[idx1].v;

            for(size_t idx2 = idx1+1; idx2 < spmat.P[i+1]; ++idx2)
            {
                size_t const j2 = spmat.X[idx2].j;
                size_t const f2 = spmat.X[idx2].f;
                double const v2 = spmat.X[idx2].v;

                double * XTr1 = 
                    XTr+j1*kNR_FIELD*nr_factor+f2*nr_factor;
                double const * w2 = 
                    W+j2*kNR_FIELD*nr_factor+f1*nr_factor;

                for(size_t d = 0; d < nr_factor; ++d, ++XTr1, ++w2)
                    *XTr1 += r1*(*w2)*v1*v2;
            }
        }
    }
}

void train(SpMat const &Tr, SpMat const &Va, Model &model, Option const &opt)
{
    FactorFunc Tr_fun(Tr, model, opt.lambda);
    TRON tron_obj(&Tr_fun, opt.eps);
    tron_obj.tron(model.W.data(), Va, model);
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
    SpMat const Tr = read_data(opt.Tr_path);
    SpMat const Va = read_data(opt.Va_path);
    printf("done\n");
    fflush(stdout);

    printf("initializing model...");
    fflush(stdout);
    Model model(Tr.nr_feature, opt.nr_factor);

    init_model(model, opt.nr_factor_real);
    printf("done\n");
    fflush(stdout);

	omp_set_num_threads(static_cast<int>(opt.nr_threads));

    train(Tr, Va, model, opt);

    if(opt.save_model)
        save_model(model, opt.model_path);

    return EXIT_SUCCESS;
}
