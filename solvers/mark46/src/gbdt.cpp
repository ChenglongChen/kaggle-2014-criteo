#include <limits>
#include <numeric>
#include <algorithm>
#include <thread>
#include <omp.h>

#include "gbdt.h"
#include "timer.h"

namespace {

float calc_bias(std::vector<float> const &Y)
{
    float y_bar = static_cast<float>(std::accumulate(Y.begin(), Y.end(), 0.0) /
        static_cast<double>(Y.size()));
    return static_cast<float>(log((1.0f+y_bar)/(1.0f-y_bar)));
}

template<typename Type>
void clean_vector(std::vector<Type> &vec)
{
    vec.clear();
    vec.shrink_to_fit();
}

void update_F(Problem &problem, CART const &tree, std::vector<float> &F)
{
    for(uint32_t i = 0; i < problem.nr_instance; ++i)
    {
        std::vector<float> x(problem.nr_field);
        for(uint32_t j = 0; j < problem.nr_field; ++j)
            x[j] = problem.X[j][i].v;
        F[i] += tree.predict(x.data()).second;
    }
}

void fit_proxy(
    TreeNode * const node,
    Problem &problem,
    std::vector<float> &F1)
{
    node->fit(problem, F1); 
}

} //unnamed namespace

uint32_t TreeNode::max_depth = 7;
uint32_t TreeNode::nr_thread = 1;
std::mutex TreeNode::mtx;
bool TreeNode::verbose = false;

void TreeNode::fit(
    Problem &problem,
    std::vector<float> &F1)
{
    if(depth >= max_depth || problem.I.size() < 100 || saturated)
    {
        double a = 0, b = 0;
        for(auto r : problem.R)
        {
            a += r;
            b += fabs(r)*(1-fabs(r));
        }
        gamma = (b <= 1e-12)? 0 : static_cast<float>(a/b);

        for(auto i : problem.I)
            F1[i] = gamma;

        is_leaf = true;

        return;
    }

    is_leaf = false;

    std::vector<std::vector<Node>> const &X = problem.X;
    std::vector<float> const &R = problem.R;

    double const sr0 = std::accumulate(R.begin(), R.end(), 0.0f);
    uint32_t const nr0 = static_cast<uint32_t>(problem.nr_instance);
    double best_ese = sr0*sr0/static_cast<double>(nr0);

    #pragma omp parallel for schedule(dynamic)
    for(uint32_t j = 0; j < problem.nr_field; ++j)
    {
        uint32_t nl = 0, nr = nr0;
        double sl = 0, sr = sr0;

        for(uint32_t i = 0; i < problem.nr_instance-1; ++i)
        {
            Node const &node = X[j][i], &node_next = X[j][i+1];
            sl += R[node.i]; 
            sr -= R[node.i]; 
            ++nl;
            --nr;
            if(node.v != node_next.v)
            {
                double const current_ese = (sl*sl)/static_cast<double>(nl) + (sr*sr)/static_cast<double>(nr);
                #pragma omp critical
                {
                    if(current_ese > best_ese || (current_ese == best_ese && static_cast<int>(j) < feature))
                    {
                        best_ese = current_ese;
                        feature = j;
                        threshold = node.v;
                    }
                }
            }
        }
    }

    if(feature == -1)
    {
        saturated = true;
        this->fit(problem, F1);
        return;
    }

    left.reset(new TreeNode(depth+1, idx*2)); 
    right.reset(new TreeNode(depth+1, idx*2+1)); 

    std::pair<Problem, Problem> sub_problems = 
        split_problem(problem, feature, threshold);

    Problem &l_problem = sub_problems.first;
    Problem &r_problem = sub_problems.second;

    if(verbose)
    {
        std::lock_guard<std::mutex> lock(mtx);
        printf("depth = %-10d   feature = %-10d   threshold = %-10.0f   left = %-10d   right = %-10d\n",
            depth, feature+1, threshold, l_problem.nr_instance, r_problem.nr_instance);
        fflush(stdout);
    }

    bool do_parallel;
    {
        std::lock_guard<std::mutex> lock(mtx);
        do_parallel = (nr_thread < 5);
    }

    if(do_parallel)
    {
        std::thread thread(fit_proxy, left.get(), std::ref(l_problem), std::ref(F1));

        {
            std::lock_guard<std::mutex> lock(mtx);
            ++nr_thread;
        }

        right->fit(r_problem, F1);
        thread.join();

        {
            std::lock_guard<std::mutex> lock(mtx);
            --nr_thread;
        }
    }
    else
    {
        left->fit(l_problem, F1);
        right->fit(r_problem, F1);
    }
}

std::pair<uint32_t, float> TreeNode::predict(float const * const x) const
{
    if(is_leaf)
        return std::make_pair(idx, gamma);
    else if(x[feature] <= threshold)
        return left->predict(x);
    else
        return right->predict(x);
}

void CART::fit(Problem &problem, std::vector<float> &F1)
{
    root.reset(new TreeNode(0, 1));

    root->fit(problem, F1);
}

std::pair<uint32_t, float> CART::predict(float const * const x) const
{
    if(!root)
        return std::make_pair(0, 0);
    else
        return root->predict(x); 
}

void GBDT::fit(Problem &Tr, Problem &Va)
{
    bias = calc_bias(Tr.Y);

    std::vector<float> F_Tr(Tr.nr_instance, bias), F_Va(Va.nr_instance, bias);

    Timer timer;
    for(uint32_t t = 0; t < trees.size(); ++t)
    {
        timer.tic();

        std::vector<float> const &Y = Tr.Y;
        std::vector<float> F1(Tr.nr_instance);

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
            Tr.R[i] = static_cast<float>(Y[i]/(1+exp(Y[i]*F_Tr[i])));
        
        trees[t].fit(Tr, F1);

        double Tr_loss = 0;
        #pragma omp parallel for schedule(static) reduction(+: Tr_loss)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
        {
            F_Tr[i] += F1[i];
            Tr_loss += log(1+exp(-Y[i]*F_Tr[i]));
        }

        printf("%3d %8.2f %10.5f", t, timer.toc(), 
            Tr_loss/static_cast<double>(Tr.Y.size()));

        if(Va.nr_instance != 0)
        {
            update_F(Va, trees[t], F_Va);

            double Va_loss = 0;
            #pragma omp parallel for schedule(static) reduction(+: Va_loss)
            for(uint32_t i = 0; i < Va.nr_instance; ++i) 
                Va_loss += log(1+exp(-Va.Y[i]*F_Va[i]));

            printf(" %10.5f", Va_loss/static_cast<double>(Va.nr_instance));
        }

        printf("\n");
        fflush(stdout);
    }
}

float GBDT::predict(float const * const x) const
{
    float s = bias;
    for(auto &tree : trees)
    {
        s += tree.predict(x).second;
    }
    return s;
}

std::vector<uint32_t> GBDT::get_indices(float const * const x) const
{
    uint32_t const nr_tree = static_cast<uint32_t>(trees.size());

    std::vector<uint32_t> indices(nr_tree);
    for(uint32_t t = 0; t < nr_tree; ++t)
        indices[t] = trees[t].predict(x).first;
    return indices;
}
