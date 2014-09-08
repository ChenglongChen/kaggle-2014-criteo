#include <vector>
#include <memory>

#include "common.h"

struct TreeNode
{
    TreeNode(size_t const depth) 
        : depth(depth), feature(-1), threshold(0), gamma(0), is_leaf(true) {}
    std::vector<size_t> I;
    size_t depth, feature;
    float threshold, gamma;
    bool is_leaf;
    std::shared_ptr<TreeNode> left, right;
    static size_t max_depth;

    void fit(
        std::vector<std::vector<float>> const &X, 
        std::vector<float> const &R, 
        std::vector<float> &F1);
    float predict(float const * const x) const;
};

class CART 
{
public:
    void fit(
        DenseColMat const &problem, 
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
    void fit(DenseColMat const &Tr, DenseColMat const &Va);

private:
    std::vector<CART> trees;
    float bias;
};
