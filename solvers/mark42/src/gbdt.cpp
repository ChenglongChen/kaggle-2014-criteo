#include <limits>
#include <numeric>
#include <algorithm>
#include <thread>
#include <omp.h>

#include "gbdt.h"
#include "timer.h"

namespace {

struct Node
{
    Node() : i(0), v(0) {}
    Node(uint64_t const i, float const v) : i(i), v(v) {}
    uint64_t i;
    float v;
};

inline double partial_sum(std::vector<float> const &R, std::vector<uint64_t> const &I)
{
    double sum = 0;
    for(auto i : I)
        sum += R[i];
    return sum;
}

inline std::vector<uint64_t> gen_init_I(uint64_t const nr_instance)
{
    std::vector<uint64_t> I(nr_instance, 0);
    for(uint64_t i = 0; i < nr_instance; ++i)
        I[i] = i;
    return I;
}

float calc_bias(std::vector<float> const &Y)
{
    float y_bar = static_cast<float>(std::accumulate(Y.begin(), Y.end(), 0.0) /
        static_cast<double>(Y.size()));
    return static_cast<float>(log((1.0f+y_bar)/(1.0f-y_bar)));
}

std::vector<Node> 
get_ordered_nodes(std::vector<float> const &Xj, std::vector<uint64_t> const &I)
{
    struct sort_by_v
    {
        bool operator() (Node const lhs, Node const rhs)
        {
            return lhs.v < rhs.v;
        }
    };

    std::vector<Node> nodes(I.size());
    for(uint64_t ii = 0; ii < I.size(); ++ii)
        nodes[ii] = Node(I[ii], Xj[I[ii]]);
    std::sort(nodes.begin(), nodes.end(), sort_by_v());

    return nodes;
}

template<typename Type>
void clean_vector(std::vector<Type> &vec)
{
    vec.clear();
    vec.shrink_to_fit();
}

void update_F(DenseColMat const &problem, CART const &tree, std::vector<float> &F)
{
    for(uint64_t i = 0; i < problem.nr_instance; ++i)
    {
        std::vector<float> x(problem.nr_field);
        for(uint64_t j = 0; j < problem.nr_field; ++j)
            x[j] = problem.X[j][i];
        F[i] += tree.predict(x.data()).second;
    }
}

void fit_proxy(
    TreeNode * const node,
    std::vector<std::vector<float>> const &X, 
    std::vector<float> const &R, 
    std::vector<float> &F1)
{
    node->fit(X, R, F1); 
}

} //unnamed namespace

uint64_t TreeNode::max_depth = 7;
uint64_t TreeNode::nr_thread = 1;
float TreeNode::alpha = 1.0f;
std::mutex TreeNode::mtx;
bool TreeNode::verbose = false;

void TreeNode::fit(
    std::vector<std::vector<float>> const &X, 
    std::vector<float> const &R, 
    std::vector<float> &F1)
{
    if(depth >= max_depth || I.size() < 100 || saturated)
    {
        double a = 0, b = 0;
        for(auto i : I)
        {
            a += R[i];
            b += fabs(R[i])*(1-fabs(R[i]));
        }
        gamma = (b <= 1e-12)? 0 : static_cast<float>(a/b);

        for(auto i : I)
            F1[i] = gamma;

        clean_vector(I);

        is_leaf = true;

        return;
    }

    is_leaf = false;

    double const sr0 = partial_sum(R, I), nr0 = static_cast<double>(I.size());
    double best_ese = sr0*sr0/nr0, base_ese = best_ese/alpha;

    #pragma omp parallel for schedule(dynamic)
    for(uint64_t j = 0; j < X.size(); ++j)
    {
        double nl = 0, nr = nr0;
        double sl = 0, sr = sr0;

        std::vector<Node> nodes = get_ordered_nodes(X[j], I);
        for(uint64_t ii = 0; ii < nodes.size()-1; ++ii)
        {
            Node const &node = nodes[ii], &node_next = nodes[ii+1];
            sl += R[node.i]; 
            sr -= R[node.i]; 
            nl += 1;
            nr -= 1;
            if(node.v != node_next.v)
            {
                double const current_ese = (sl*sl)/nl + (sr*sr)/nr;
                #pragma omp critical
                {
                    if(current_ese > base_ese && (current_ese > best_ese || 
                        (current_ese == best_ese && static_cast<int>(j) < feature)))
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
        this->fit(X, R, F1);
        return;
    }

    left.reset(new TreeNode(depth+1, idx*2)); 
    right.reset(new TreeNode(depth+1, idx*2+1)); 

    for(auto i : I)
    {
        if(X[feature][i] <= threshold)
            left->I.push_back(i);
        else
            right->I.push_back(i);
    }

    clean_vector(I);

    if(verbose)
    {
        std::lock_guard<std::mutex> lock(mtx);
        printf("depth = %-10ld   feature = %-10ld   threshold = %-10.0f   left = %-10ld   right = %-10ld\n",
            depth, feature+1, threshold, left->I.size(), right->I.size());
        fflush(stdout);
    }

    bool do_parallel;
    {
        std::lock_guard<std::mutex> lock(mtx);
        do_parallel = (nr_thread < 5);
    }

    if(do_parallel)
    {
        std::thread thread(fit_proxy, left.get(), std::ref(X), std::ref(R), std::ref(F1));

        {
            std::lock_guard<std::mutex> lock(mtx);
            ++nr_thread;
        }

        right->fit(X, R, F1);
        thread.join();

        {
            std::lock_guard<std::mutex> lock(mtx);
            --nr_thread;
        }
    }
    else
    {
        left->fit(X, R, F1);
        right->fit(X, R, F1);
    }
}

std::pair<uint64_t, float> TreeNode::predict(float const * const x) const
{
    if(is_leaf)
        return std::make_pair(idx, gamma);
    else if(x[feature] <= threshold)
        return left->predict(x);
    else
        return right->predict(x);
}

void CART::fit(DenseColMat const &problem, std::vector<float> const &R, std::vector<float> &F1)
{
    root.reset(new TreeNode(0, 1));
    root->I = gen_init_I(problem.nr_instance);

    root->fit(problem.X, R, F1);
}

std::pair<uint64_t, float> CART::predict(float const * const x) const
{
    if(!root)
        return std::make_pair(0, 0);
    else
        return root->predict(x); 
}

void GBDT::fit(DenseColMat const &Tr, DenseColMat const &Va)
{
    bias = calc_bias(Tr.Y);

    std::vector<float> F_Tr(Tr.nr_instance, bias), F_Va(Va.nr_instance, bias);

    Timer timer;
    for(uint64_t t = 0; t < trees.size(); ++t)
    {
        timer.tic();

        std::vector<float> const &Y = Tr.Y;
        std::vector<float> R(Tr.nr_instance), F1(Tr.nr_instance);

        #pragma omp parallel for schedule(static)
        for(uint64_t i = 0; i < Tr.nr_instance; ++i) 
            R[i] = static_cast<float>(Y[i]/(1+exp(Y[i]*F_Tr[i])));
        
        trees[t].fit(Tr, R, F1);

        double Tr_loss = 0;
        #pragma omp parallel for schedule(static) reduction(+: Tr_loss)
        for(uint64_t i = 0; i < Tr.nr_instance; ++i) 
        {
            F_Tr[i] += F1[i];
            Tr_loss += log(1+exp(-Y[i]*F_Tr[i]));
        }

        printf("%3ld %8.2f %10.5f", t, timer.toc(), 
            Tr_loss/static_cast<double>(Tr.Y.size()));

        if(Va.nr_instance != 0)
        {
            update_F(Va, trees[t], F_Va);

            double Va_loss = 0;
            #pragma omp parallel for schedule(static) reduction(+: Va_loss)
            for(uint64_t i = 0; i < Va.nr_instance; ++i) 
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

std::vector<uint64_t> GBDT::get_indices(float const * const x) const
{
    uint64_t const nr_tree = trees.size();

    std::vector<uint64_t> indices(nr_tree);
    for(uint64_t t = 0; t < nr_tree; ++t)
        indices[t] = trees[t].predict(x).first;
    return indices;
}
