#include <limits>
#include <numeric>
#include <algorithm>

#include "gbdt.h"

namespace {

struct Node
{
    Node() : i(0), v(0) {}
    Node(size_t const i, float const v) : i(i), v(v) {}
    size_t i;
    float v;
};

inline double calc_ese(std::vector<float> const &R, std::vector<size_t> const &I)
{
    double ese = 0;
    for(auto i : I)
        ese += R[i];
    return ese*ese/static_cast<double>(I.size());
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
    for(auto i : I)
        nodes[i] = Node(i, Xj[i]);
    std::sort(nodes.begin(), nodes.end(), sort_by_v());

    return nodes;
}

template<typename Type>
void clean_vector(std::vector<Type> &vec)
{
    vec.clear();
    vec.shrink_to_fit();
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

    double best_ese = std::numeric_limits<double>::max();
    for(size_t j = 0; j < kNR_FEATURE; ++j)
    {
        double nl = 0, nr = static_cast<double>(I.size());
        double sl = 0, sr = calc_ese(R, I);

        std::vector<Node> nodes = get_ordered_nodes(X[j], I);
        for(size_t ii = 0; ii < nodes.size()-1; ++ii)
        {
            Node const &node = nodes[ii], &node_next = nodes[ii+1];
            sl += R[node.i]; 
            sr -= R[node.i]; 
            --nl;
            ++nr;
            if(node.i != node_next.i)
            {
                double const current_ese = sl*sl/nl + sr*sr/nr;
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
        if(X[feature][i] < threshold)
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

void CART::fit(Problem const &problem, std::vector<float> const &R, std::vector<float> F1)
{
    root.reset(new TreeNode);
    root->I = gen_init_I(problem.nr_instance);

    size_t nr_leaf = 1;
    root->fit(problem.X, R, F1, nr_leaf);
}

void GBDT::fit(Problem const &problem)
{
    size_t const nr_instance = problem.nr_instance;

    bias = calc_bias(problem.Y);

    std::vector<float> F(nr_instance, bias);

    for(auto &tree : trees)
    {
        std::vector<float> const &Y = problem.Y;
        std::vector<float> R(nr_instance), F1(nr_instance);

        for(size_t i = 0; i < nr_instance; ++i) 
            R[i] = static_cast<float>(Y[i]/(1+exp(Y[i]*F[i])));
        
        tree.fit(problem, R, F1);

        for(size_t i = 0; i < nr_instance; ++i) 
            F[i] += F1[i];
    }
}
