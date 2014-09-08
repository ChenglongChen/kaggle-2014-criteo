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

} //unnamed namespace

void TreeNode::fit(std::vector<std::vector<Node>> const &X, std::vector<float> const &R)
{
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

    left->fit();
    right->fit();
}

void CART::fit(Problem const &problem, std::vector<float> const &R)
{
    root.reset(new TreeNode);
    root->II = gen_init_II(problem.nr_instance);
    root->fit(problem.X, R);
}

void GBDT::fit(Problem &problem)
{
    // sort problem
    for(auto &tree : trees)
    {
        // calc R
        tree.fit(problem, R);
    }
    // inverse sort problem
}
