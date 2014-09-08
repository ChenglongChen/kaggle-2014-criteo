#include <limits>

namespace {

inline double calc_ese(std::vector<float> const &R)
{
    return pow(std::accumulate(R.begin(), R.end(), 0.0), 2)/static_cast<double>(R.size());
}

inline std::vector<size_t> gen_init_II(size_t const nr_instance)
{
    std::vector<size_t> II(nr_instance, 0);
    for(size_t ii = 0; ii < nr_instance; ++ii)
        II[ii] = ii;
    return II;
}

void sort_problem_by_v(Problem &problem)
{
    struct sort_by_v
    {
        bool operator() (Node const lhs, Node const rhs)
        {
            return lhs.v < rhs.v;
        }
    };

    for(size_t j = 0; j < kNR_FEATURE; ++j)
        std::sort(problem.X[j].begin(), problem.X[j].end(), sort_by_v());
}

void sort_problem_by_i(Problem &problem)
{
    struct sort_by_i
    {
        bool operator() (Node const lhs, Node const rhs)
        {
            return lhs.i < rhs.i;
        }
    };

    for(size_t j = 0; j < kNR_FEATURE; ++j)
        std::sort(problem.X[j].begin(), problem.X[j].end(), sort_by_i());
}

inline float logistic_func(float const s)
{
    return 1/(1+static_cast<float>(exp(-s)));
}

} //unnamed namespace

void TreeNode::fit(std::vector<std::vector<Node>> const &X, std::vector<float> const &R, std::vector<float> &F1, size_t &nr_leaf)
{
    if(nr_leaf >= kMAX_NR_LEAF)
    {
        double a = 0, b = 0;
        for(auto ii = II.begin(); ii != II.end()-1; ++ii)
        {
            Node const &node = X[j][*ii];
            a += R[node.i];
            b += fabs(R[node.i])*(1-fabs(R[node.i]));
        }
        gamma = a/b;

        for(auto ii = II.begin(); ii != II.end()-1; ++ii)
        {
            Node const &node = X[j][*ii];
            F1[node.i] = gamma;
        }

        return;
    }

    nr_leaf += 2;
    is_leaf = false;

    std::vector<std::vector<Node>> const &X = problem.X;

    threshold = 0, feature = 0;
    double base_ese = calc_ese(R), best_ese = base_ese;
    for(size_t j = 0; j < kNR_FEATURE; ++j)
    {
        double nl = 0, nr = static_cast<double>(R.size());
        double sl = 0, sr = std::accumulate(R.begin(), R.end(), 0.0f);
        for(size_t j = 1; j < kNR_FEATURE; ++j)
        {
            for(auto ii = II.begin(); ii != II.end()-1; ++ii)
            {
                Node const &node = X[j][*ii], &node_next = X[j][*(ii+1)];
                sl += R[node.i]; 
                sr -= R[node.i]; 
                if(node.i != node_next.i)
                {
                    double const current_ese = sl*sl/nl + sr*sr/nr;
                    if(current_ese > best_ese)
                    {
                        best_ese = current_ese;
                        feat = j;
                        threshold = node.v;
                    }
                }
            }
        }
    }

    left.reset(new TreeNode); 
    right.reset(new TreeNode); 

    for(auto i : I)
    {
        if(problem.X[feature][i] < threshold)
            left->I.push_back(i);
        else
            right->I.push_back(i);
    }

    I.clear();

    if(left->I.size() > right->I.size())
    {
        left->fit(X, R, nr_leaf);
        right->fit(X, R, nr_leaf);
    }
    else
    {
        right->fit(X, R, nr_leaf);
        left->fit(X, R, nr_leaf);
    }
}

void CART::fit(Problem const &problem, std::vector<float> const &R, std::vector<float> F1)
{
    root.reset(new TreeNode);
    root->II = gen_init_II(problem.nr_instance);

    size_t nr_leaf = 1;
    root->fit(problem.X, R, nr_leaf);
}

void GBDT::fit(Problem &problem)
{
    size_t const nr_instance = problem.nr_instance;
    std::vector<float> &Y = problem.Y;

    calc_bias(problem);

    std::vector<float> F(nr_instance, bias);

    sort_problem_by_v(problem);
    for(auto &tree : trees)
    {
        std::vector<float> R(nr_instance), F1(nr_instance);
        for(size_t i = 0; i < nr_instance; ++i) 
            R[i] = Y[i]/(1+exp(Y[i]*F[i]));
        
        tree.fit(problem, R, F1);
    }
    sort_problem_by_i(problem);
}

void GBDT::calc_bias(std::vector<float> &Y)
{
    float y_bar = static_cast<float>(std::accumulate(Y.begin(), Y.end(), 0.0) /
        static_cast<double>(Y.size()));
    return log((1.0f+y_bar)/(1.0f-y_bar));
}
