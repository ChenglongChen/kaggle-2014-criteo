#include <stdexcept>
#include <cstring>
#include <cassert>
#include <algorithm>

#include "common.h"

namespace {

uint32_t const kMaxLineSize = 1000000;

uint32_t get_nr_line(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    uint32_t nr_line = 0;
    while(fgets(line, kMaxLineSize, f) != nullptr)
        ++nr_line;

    fclose(f);

    return nr_line;
}

uint32_t get_nr_field(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    fgets(line, kMaxLineSize, f);
    strtok(line, " \t");

    uint32_t nr_field = 0;
    while(1)
    {
        char *val_char = strtok(nullptr," \t");
        if(val_char == nullptr || *val_char == '\n')
            break;
        ++nr_field;
    }

    fclose(f);

    return nr_field;
}

void read_dcm(Problem &problem, std::string const &path, bool const do_sort)
{
    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");
    for(uint32_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        char *p = strtok(line, " \t");
        problem.Y[i] = (atoi(p)>0)? 1.0f : -1.0f;
        for(uint32_t j = 0; j < problem.nr_field; ++j)
        {
            char *val_char = strtok(nullptr," \t");

            float const val = static_cast<float>(atof(val_char));

            problem.X[j][i] = Node(i, val);
        }
    }

    if(do_sort)
    {
        struct sort_by_v
        {
            bool operator() (Node const lhs, Node const rhs)
            {
                return lhs.v < rhs.v;
            }
        };

        for(uint32_t j = 0; j < problem.nr_field; ++j)
        {
            std::vector<Node> &X1 = problem.X[j];
            std::vector<Node> &Z1 = problem.Z[j];
            std::sort(X1.begin(), X1.end(), sort_by_v());
            for(uint32_t i = 0; i < problem.nr_instance; ++i)
                Z1[X1[i].i] = Node(i, X1[i].v);
        }
    }

    fclose(f);
}

void read_scm(Problem &problem, std::string const &path)
{
    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");

    std::vector<std::vector<uint32_t>> buffer;

    uint64_t nnz = 0; 
    uint32_t nr_instance = 0;
    for(; fgets(line, kMaxLineSize, f) != nullptr; ++nr_instance)
    {
        strtok(line, " \t");
        for( ; ; ++nnz)
        {
            char *idx_char = strtok(nullptr," \t");
            if(idx_char == nullptr || *idx_char == '\n')
                break;

            uint32_t const idx = atoi(idx_char);

            buffer.resize(idx);
            buffer[idx-1].push_back(nr_instance);
        }
    }

    problem.nr_sparse_field = static_cast<uint32_t>(buffer.size());
    problem.SX.resize(nr_instance);
    problem.SP.resize(problem.nr_sparse_field+1);
    problem.SP[0] = 0;

    uint64_t p = 0;
    for(uint32_t j = 0; j < problem.nr_sparse_field; ++j)
    {
        for(auto i : buffer[j]) 
            problem.SX[p++] = i;
        problem.SP[j+1] = p;
    }

    fclose(f);
}

} //unamed namespace

std::pair<Problem, Problem> split_problem(Problem const &problem, 
    uint32_t const feature, float const threshold)
{
    uint32_t const nr_instance = problem.nr_instance;
    uint32_t const nr_field = problem.nr_field;

    std::vector<int> partition(nr_instance, 0);
    uint32_t l_size = 0, r_size = 0;
    for(uint32_t i = 0; i < nr_instance; ++i)
    {
        float const v = problem.Z[feature][i].v;
        if(v <= threshold)
        {
            partition[i] = -1;
            ++l_size;
        }
        else
        {
            partition[i] = 1;
            ++r_size;
        }
    }

    Problem l_problem(l_size, nr_field), r_problem(r_size, nr_field);
    #pragma omp parallel for schedule(dynamic)
    for(uint32_t j = 0; j < nr_field; ++j)
    {
        std::vector<Node> const &X1 = problem.X[j];
        std::vector<Node> const &Z1 = problem.Z[j];

        std::vector<int32_t> partition1(nr_instance, 0);
        std::vector<uint32_t> bridge(nr_instance, 0);
        for(uint32_t i = 0, l_i = 0, r_i = 0; i < nr_instance; ++i)
        {
            if(partition[i] == -1)
            {
                l_problem.Z[j][l_i] = Z1[i];
                bridge[i] = l_i++;
            }
            else if(partition[i] == 1)
            {
                r_problem.Z[j][r_i] = Z1[i];
                bridge[i] = r_i++;
            }
            else 
            {
                assert(false);
            }
            partition1[Z1[i].i] = partition[i];
        }

        for(uint32_t i = 0, l_i = 0, r_i = 0; i < nr_instance; ++i)
        {
            uint32_t const new_i = bridge[X1[i].i];
            if(partition1[i] == -1)
            {
                l_problem.Z[j][new_i].i = l_i;
                l_problem.X[j][l_i] = Node(new_i, X1[i].v);
                ++l_i;
            }
            else if(partition1[i] == 1)
            {
                r_problem.Z[j][new_i].i = r_i;
                r_problem.X[j][r_i] = Node(new_i, X1[i].v);
                ++r_i;
            }
            else 
            {
                assert(false);
            }
        }
    }

    for(uint32_t i = 0, l_i = 0, r_i = 0; i < nr_instance; ++i)
    {
        if(partition[i] == -1)
        {
            l_problem.Y[l_i] = problem.Y[i];
            l_problem.R[l_i] = problem.R[i];
            l_problem.I[l_i++] = problem.I[i];
        }
        else if(partition[i] == 1)
        {
            r_problem.Y[r_i] = problem.Y[i];
            r_problem.R[r_i] = problem.R[i];
            r_problem.I[r_i++] = problem.I[i];
        }
        else 
        {
            assert(false);
        }
    }
    
    return std::make_pair(l_problem, r_problem);
}

Problem read_data(std::string const &dense_path, std::string const &sparse_path,
    bool const do_sort)
{
    Problem problem(get_nr_line(dense_path), get_nr_field(dense_path));

    read_dcm(problem, dense_path, do_sort);

    read_scm(problem, sparse_path);

    return problem;
}

FILE *open_c_file(std::string const &path, std::string const &mode)
{
    FILE *f = fopen(path.c_str(), mode.c_str());
    if(!f)
        throw std::runtime_error(std::string("cannot open ")+path);
    return f;
}

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}
