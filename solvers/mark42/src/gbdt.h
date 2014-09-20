#include <vector>
#include <memory>
#include <mutex>

#include "common.h"

struct TreeNode
{
    TreeNode(uint64_t const depth, uint64_t const idx) 
        : depth(depth), idx(idx), feature(-1), threshold(0), gamma(0),
          is_leaf(true), saturated(false) {}
    std::vector<uint64_t> I;
    uint64_t depth, idx;
    int64_t feature;
    float threshold, gamma;
    bool is_leaf, saturated;
    std::shared_ptr<TreeNode> left, right;
    static uint64_t max_depth, nr_thread;
    static std::mutex mtx;
    static bool verbose;
    static float alpha;

    void fit(
        std::vector<std::vector<float>> const &X, 
        std::vector<float> const &R, 
        std::vector<float> &F1);
    std::pair<uint64_t, float> predict(float const * const x) const;
};

class CART 
{
public:
    void fit(
        DenseColMat const &problem, 
        std::vector<float> const &R, 
        std::vector<float> &F1);
    std::pair<uint64_t, float> predict(float const * const x) const;

private:
    std::shared_ptr<TreeNode> root;
};

class GBDT
{
public:
    GBDT(uint64_t const nr_tree) : trees(nr_tree), bias(0) {}
    void fit(DenseColMat const &Tr, DenseColMat const &Va);
    float predict(float const * const x) const;
    std::vector<uint64_t> get_indices(float const * const x) const;

private:
    std::vector<CART> trees;
    float bias;
};
