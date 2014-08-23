#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <omp.h>

#include "common.h"
#include "timer.h"

namespace {

inline float solve_z(
    float const * const Y,
    float const * const S,
    float const * const A,
    float const z_init,
    float const lambda,
    size_t const nr_instance)
{
    if(nr_instance == 0)
        return z_init;
	float z = z_init;
	float z_new = 0;
	float f = 0;
	float f_new = 0;
	float g = 0;
	float h = 0;
	float d = 0;
	float exp_dec = 0;
	const float beta = 0.5;
	const float gamma = 0.1f;
	const size_t max_iter = 2;

	for(size_t t = 1; t <= max_iter; t++){
		f = lambda / 2 * z * z;
		g = lambda*z;
		h = lambda;
		for(size_t i = 0; i <= nr_instance - 1; i++){
			exp_dec = std::exp(-Y[i] * (S[i] + z * A[i]));
			f += std::log(1 + exp_dec);
			g += -Y[i] * A[i] * exp_dec / (1 + exp_dec);
			h += exp_dec * static_cast<float>(pow(A[i] / (1 + exp_dec), 2));
		}
		d = -g / h;
		
        size_t nr_line_search = 0;
		do{
			z_new = z + d;
			f_new = lambda / 2 * z_new * z_new;
			for(size_t i = 0; i <= nr_instance - 1; i++)
				f_new += std::log( 1 + std::exp(-Y[i] * (S[i] + z_new * A[i])));
			d *= beta;
            ++nr_line_search;
            if(nr_line_search > 10)
                break;
		}while(f_new - f > gamma * d  * g);
	}
	return static_cast<float>(z_new);
}

struct AuxiliaryMat
{
    AuxiliaryMat(size_t const nr_instance) : nr_instance(nr_instance) {}
    std::vector<size_t> P, I;
    std::vector<float> Y, S, A;
    size_t const nr_instance;
};

std::vector<AuxiliaryMat> get_aux_matrices(SpMat const &spmat)
{
    std::vector<AuxiliaryMat> aux_mats(spmat.nr_field, spmat.nr_instance);
    for(size_t f = 0; f < spmat.nr_field; ++f)
    {
        size_t const nr_feature = spmat.nr_field_feature[f];
        std::vector<std::vector<std::pair<size_t, float>>> buff(nr_feature);
        for(size_t i = 0; i < spmat.nr_instance; ++i)
        {
            Node const &x = spmat.X[spmat.P[i]+f];
            if(x.j >= nr_feature)
                continue;
            buff[x.j].emplace_back(i, spmat.Y[i]); 
        }

        AuxiliaryMat &aux_mat = aux_mats[f];
        aux_mat.P.push_back(0);
        for(size_t j = 0; j < nr_feature; ++j)
        {
            for(auto &tuple : buff[j])
            {
                aux_mat.I.push_back(tuple.first);
                aux_mat.Y.push_back(tuple.second);
                aux_mat.S.push_back(0);
                aux_mat.A.push_back(0);
            }
            aux_mat.P.push_back(aux_mat.I.size());
        }
    }
    return aux_mats;
}

struct Option
{
    Option() : lambda(2.0f), iter(100), inner_iter(2), nr_factor(1), nr_thread(1), save_model(true) {}
    std::string Tr_path, model_path, Va_path;
    float lambda;
    size_t iter, inner_iter, nr_factor, nr_thread;
    bool save_model;
};

std::string train_help()
{
    return std::string(
"usage: sgd-poly2-train [<options>] <train_path>\n"
"\n"
"options:\n"
"-l <lambda>: you know\n"
"-k <dimension>: you know\n"
"-T <iteration>: you know\n"
"-t <iteration>: you know\n"
"-s <nr_thread>: you know\n"
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
        if(args[i].compare("-T") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.iter = std::stoi(args[++i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.inner_iter = std::stoi(args[++i]);
        }
        else if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_factor = std::stoi(args[++i]);
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
            opt.nr_thread = std::stoi(args[++i]);
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

void init_model(Model &model)
{
    size_t const nr_factor = model.nr_factor;
    float const coef = 
        static_cast<float>(0.5/sqrt(static_cast<double>(nr_factor)));

    for(size_t f = 0; f < model.nr_field; ++f)
    {
        float * w = model.W[f].data();
        for(size_t j = 0; j < model.nr_field_feature[f]; ++j)
            for(size_t f = 0; f < model.nr_field; ++f)
                for(size_t d = 0; d < nr_factor; ++d, ++w)
                    *w = coef*static_cast<float>(drand48());
    }
}

void update_s(SpMat const &spmat, Model const &model, std::vector<float> &S,
    size_t const d, size_t const f1, size_t const f2, bool const do_addition)
{
    size_t const nr_field = model.nr_field;
    size_t const nr_factor = model.nr_factor;

    for(size_t i = 0; i < spmat.nr_instance; ++i)
    {
        Node const * const x = &spmat.X[spmat.P[i]];
        Node const * const x1 = x+f1;
        Node const * const x2 = x+f2;

        size_t const j1 = x1->j;
        if(j1 >= model.nr_field_feature[f1])
            continue;
        size_t const j2 = x2->j;
        if(j2 >= model.nr_field_feature[f2])
            continue;

        float const v1 = x1->v;
        float const v2 = x2->v;

        float const &w1 = 
            model.W[f1][j1*nr_field*nr_factor+f2*nr_factor+d];
        float const &w2 = 
            model.W[f2][j2*nr_field*nr_factor+f1*nr_factor+d];

        float const delta = w1*w2*v1*v2;

        if(do_addition)
            S[i] += delta;
        else
            S[i] -= delta;
    }
}

void update_w(
    SpMat const &spmat, 
    Model &model, 
    std::vector<float> const &S,
    size_t const d,
    size_t const f1,
    size_t const f2,
    std::vector<AuxiliaryMat> &aux_mats,
    float const lambda)
{
    size_t const nr_field = model.nr_field;
    size_t const nr_factor = model.nr_factor;

    for(size_t j = 0; j < model.nr_field_feature[f1]; ++j)
    {
        AuxiliaryMat &aux_mat = aux_mats[f1];
        for(size_t idx = aux_mat.P[j]; idx < aux_mat.P[j+1]; ++idx)
        {
            size_t const i = aux_mat.I[idx];
            Node const &x1 = spmat.X[spmat.P[i]+f1];
            Node const &x2 = spmat.X[spmat.P[i]+f2];

            aux_mat.S[idx] = S[i];
            aux_mat.A[idx] = model.W[f2][x2.j*nr_field*nr_factor+x1.f*nr_factor+d]*x1.v*x2.v;
        }

        float &z = model.W[f1][j*nr_field*nr_factor+f2*nr_factor+d];
        size_t const idx = aux_mat.P[j];
        z = solve_z(&aux_mat.Y[idx], &aux_mat.S[idx], &aux_mat.A[idx], z, lambda, aux_mat.P[j+1]-aux_mat.P[j]);
    }
}

void train(SpMat const &Tr, SpMat const &Va, Model &model, Option const &opt)
{
    size_t const nr_field = model.nr_field;
    size_t const nr_factor = model.nr_factor;

    std::vector<AuxiliaryMat> Tr_aux = get_aux_matrices(Tr);
    std::vector<float> Tr_S = calc_s(Tr, model);
    std::vector<float> Va_S = calc_s(Va, model);

    Timer timer;
    for(size_t t = 0; t < opt.iter; ++t)
    {
        for(size_t d = 0; d < nr_factor; ++d)
        {
            for(size_t f1 = 0; f1 < nr_field; ++f1)
            {
                for(size_t f2 = f1+1; f2 < nr_field; ++f2)
                {
                    timer.tic();
                    update_s(Tr, model, Tr_S, d, f1, f2, false);
                    update_s(Va, model, Va_S, d, f1, f2, false);

                    for(size_t tt = 0; tt < opt.inner_iter; ++tt)
                    {
                        update_w(Tr, model, Tr_S, d, f1, f2, Tr_aux, opt.lambda);
                        update_w(Tr, model, Tr_S, d, f2, f1, Tr_aux, opt.lambda);
                    }

                    update_s(Tr, model, Tr_S, d, f1, f2, true);
                    update_s(Va, model, Va_S, d, f1, f2, true);

                    printf("%3ld %3ld %3ld %3ld %8.2f %10.5f", 
                        t, d, f1+1, f2+1, timer.toc(), calc_loss(Tr.Y, Tr_S));
                    if(Va.Y.size() != 0)
                        printf(" %10.5f", calc_loss(Va.Y, Va_S));
                    printf("\n");
                    fflush(stdout);
                }
            }
        }
    }
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
    Model model(Tr.nr_field, opt.nr_factor, Tr.nr_field_feature);

    init_model(model);
    printf("done\n");
    fflush(stdout);

	//omp_set_num_threads(static_cast<int>(opt.nr_thread));

    train(Tr, Va, model, opt);

    if(opt.save_model)
        save_model(model, opt.model_path);

    return EXIT_SUCCESS;
}
