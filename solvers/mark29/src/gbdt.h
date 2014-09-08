#include <vector>
#include <memory>

#include "common.h"

struct TreeNode
{
    TreeNode(size_t const depth, size_t const idx) 
        : depth(depth), idx(idx), feature(-1), threshold(0), gamma(0),
          is_leaf(true), saturated(false) {}
    std::vector<size_t> I;
    size_t depth, idx;
    int64_t feature;
    float threshold, gamma;
    bool is_leaf, saturated;
    std::shared_ptr<TreeNode> left, right;
    static size_t max_depth;

    void fit(
        std::vector<std::vector<float>> const &X, 
        std::vector<float> const &R, 
        std::vector<float> &F1);
    std::pair<size_t, float> predict(float const * const x) const;
};

class CART 
{
public:
    void fit(
        DenseColMat const &problem, 
        std::vector<float> const &R, 
        std::vector<float> &F1);
    std::pair<size_t, float> predict(float const * const x) const;

private:
    std::shared_ptr<TreeNode> root;
};

class GBDT
{
public:
    GBDT(size_t const nr_tree) : trees(nr_tree), bias(0) {}
    void fit(DenseColMat const &Tr, DenseColMat const &Va);
    float predict(float const * const x) const;
    std::vector<size_t> get_indices(float const * const x) const;

private:
    std::vector<CART> trees;
    float bias;
};