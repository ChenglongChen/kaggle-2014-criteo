
void TreeNode::split(Problem const &problem, float const threshold)
{
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
}

void CART::fit(Problem const &problem, std::vector<float> const &R)
{
    size_t nl = 0, nr = R.size();
    size_t sl = 0, sr = std::accumulate(R.begin(), R.end(), 0.0f);
    size_t best_idx = 0, best_val = sr*sr/nr;
    for(size_t j = 0; j < kNR_FEATURE; ++j)
    {
        for(auto &node : problem.X[j])
        for(size_t i = 0, )
        {
            sl += R[node.i]; 
            sr -= R[node.i]; 
            if()
        }
    }
}
