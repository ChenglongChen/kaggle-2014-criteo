#include <limits>
#include <numeric>
#include <algorithm>

#include "gbdt.h"
#include "timer.h"

namespace {

struct Node
{
    Node() : i(0), v(0) {}
    Node(size_t const i, float const v) : i(i), v(v) {}
    size_t i;
    float v;
};

inline double partial_sum(std::vector<float> const &R, std::vector<size_t> const &I)
{
    double sum = 0;
    for(auto i : I)
        sum += R[i];
    return sum;
}

inline std::vector<size_t> gen_init_I(size_t const nr_instance)
{
    std::vector<size_t> I(nr_instance, 0);
    for(size_t i = 0; i < nr_instance; ++i)
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
get_ordered_nodes(std::vector<float> const &Xj, std::vector<size_t> const &I)
{
    struct sort_by_v
    {
        bool operator() (Node const lhs, Node const rhs)
        {
            return lhs.v < rhs.v;
        }
    };

    std::vector<Node> nodes(I.size());
    for(size_t ii = 0; ii < I.size(); ++ii)
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

void update_F(Problem const &problem, CART const &tree, std::vector<float> &F)
{
    for(size_t i = 0; i < problem.nr_instance; ++i)
    {
        std::vector<float> x(kNR_FEATURE);
        for(size_t j = 0; j < kNR_FEATURE; ++j)
            x[j] = problem.X[j][i];
        F[i] += tree.predict(x.data());
    }
}

} //unnamed namespace

void TreeNode::fit(
    std::vector<std::vector<float>> const &X, 
    std::vector<float> const &R, 
    std::vector<float> &F1, 
    size_t &nr_leaf)
{
    if(nr_leaf >= kMAX_NR_LEAF)
    {
        double a = 0, b = 0;
        for(auto i : I)
        {
            a += R[i];
            b += fabs(R[i])*(1-fabs(R[i]));
        }
        gamma = static_cast<float>(a/b);

        for(auto i : I)
            F1[i] = gamma;

        clean_vector(I);

        return;
    }

    nr_leaf += 2;
    is_leaf = false;

    double best_ese = 0;
    for(size_t j = 0; j < kNR_FEATURE; ++j)
    {
        double nl = 0, nr = static_cast<double>(I.size());
        double sl = 0, sr = partial_sum(R, I);

        std::vector<Node> nodes = get_ordered_nodes(X[j], I);
        for(size_t ii = 0; ii < nodes.size()-1; ++ii)
        {
            Node const &node = nodes[ii], &node_next = nodes[ii+1];
            sl += R[node.i]; 
            sr -= R[node.i]; 
            nl += 1;
            nr -= 1;
            if(node.v != node_next.v)
            {
                double const current_ese = (sl*sl)/nl + (sr*sr)/nr;
                if(current_ese > best_ese)
                {
                    best_ese = current_ese;
                    feature = j;
                    threshold = node.v;
                }
            }
        }
    }

    left.reset(new TreeNode); 
    right.reset(new TreeNode); 

    for(auto i : I)
    {
        if(X[feature][i] <= threshold)
            left->I.push_back(i);
        else
            right->I.push_back(i);
    }

    clean_vector(I);

    if(left->I.size() > right->I.size())
    {
        left->fit(X, R, F1, nr_leaf);
        right->fit(X, R, F1, nr_leaf);
    }
    else
    {
        right->fit(X, R, F1, nr_leaf);
        left->fit(X, R, F1, nr_leaf);
    }
}

float TreeNode::predict(float const * const x) const
{
    if(is_leaf)
        return gamma;
    else if(x[feature] <= threshold)
        return left->predict(x);
    else
        return right->predict(x);
}

void CART::fit(Problem const &problem, std::vector<float> const &R, std::vector<float> &F1)
{
    root.reset(new TreeNode);
    root->I = gen_init_I(problem.nr_instance);

    size_t nr_leaf = 1;
    root->fit(problem.X, R, F1, nr_leaf);
}

float CART::predict(float const * const x) const
{
    if(!root)
        return 0;
    else
        return root->predict(x); 
}

void GBDT::fit(Problem const &Tr, Problem const &Va)
{
    bias = calc_bias(Tr.Y);

    std::vector<float> F_Tr(Tr.nr_instance, bias), F_Va(Va.nr_instance, bias);

    Timer timer;
    for(size_t t = 0; t < trees.size(); ++t)
    {
        timer.tic();

        std::vector<float> const &Y = Tr.Y;
        std::vector<float> R(Tr.nr_instance), F1(Tr.nr_instance);

        for(size_t i = 0; i < Tr.nr_instance; ++i) 
            R[i] = static_cast<float>(Y[i]/(1+exp(Y[i]*F_Tr[i])));
        
        trees[t].fit(Tr, R, F1);

        double Tr_loss = 0;
        for(size_t i = 0; i < Tr.nr_instance; ++i) 
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
            for(size_t i = 0; i < Va.nr_instance; ++i) 
                Va_loss += log(1+exp(-Va.Y[i]*F_Va[i]));

            printf(" %10.5f", Va_loss/static_cast<double>(Va.nr_instance));
        }

        printf("\n");
        fflush(stdout);
    }
}
