#include <vector>

#include "common.h"

size_t const kMAX_NR_LEAF;

struct TreeNode
{
    TreeNode() : is_leaf(true) {}
    std::vector<size_t> I;
    size_t feature;
    float threshold, gamma;
    bool is_leaf;
    std::shared_ptr<TreeNode> left, right;

    void fit(std::vector<std::vector<Node>> const &X, std::vector<float> const &R, std::vector<float> &F1, size_t &nr_leaf)
};

class CART 
{
    void fit(Problem const &problem, std::vector<float> const &R, std::vector<float> F1)

private:
    std::shared_ptr<TreeNode> root;
};

class GBDT
{
    GBDT(size_t const nr_tree) : trees(nr_tree), bias(0) {}
    void fit(Problem const &problem);

private:
    void calc_bias(Problem const &problem);

    std::vector<CART> trees;
    float bias;
};
