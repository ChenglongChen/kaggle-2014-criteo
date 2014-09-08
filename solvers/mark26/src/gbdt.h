#include <vector>
#include <memory>

#include "common.h"

size_t const kMAX_NR_LEAF = 10;

struct TreeNode
{
    TreeNode() : feature(-1), threshold(0), gamma(0), is_leaf(true) {}
    std::vector<size_t> I;
    size_t feature;
    float threshold, gamma;
    bool is_leaf;
    std::shared_ptr<TreeNode> left, right;

    void fit(
        std::vector<std::vector<float>> const &X, 
        std::vector<float> const &R, 
        std::vector<float> &F1, 
        size_t &nr_leaf);
    float predict(float const * const x) const;
};

class CART 
{
public:
    void fit(
        Problem const &problem, 
        std::vector<float> const &R, 
        std::vector<float> &F1);
    float predict(float const * const x) const;

private:
    std::shared_ptr<TreeNode> root;
};

class GBDT
{
public:
    GBDT(size_t const nr_tree) : trees(nr_tree), bias(0) {}
    void fit(Problem const &Tr, Problem const &Va);

private:
    std::vector<CART> trees;
    float bias;
};
