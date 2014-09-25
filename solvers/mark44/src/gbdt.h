#include <vector>
#include <memory>
#include <mutex>

#include "common.h"

struct TreeNode
{
    TreeNode(uint32_t const depth, uint32_t const idx) 
        : depth(depth), idx(idx), feature(-1), threshold(0), gamma(0),
          is_leaf(true), saturated(false) {}
    std::vector<uint32_t> I;
    uint32_t depth, idx;
    int32_t feature;
    float threshold, gamma;
    bool is_leaf, saturated;
    std::shared_ptr<TreeNode> left, right;
    static uint32_t max_depth, nr_thread;
    static std::mutex mtx;
    static bool verbose;

    void fit(
        std::vector<std::vector<float>> const &X, 
        std::vector<float> const &R, 
        std::vector<float> &F1);
    std::pair<uint32_t, float> predict(float const * const x) const;
};

class CART 
{
public:
    void fit(
        Problem const &problem, 
        std::vector<float> const &R, 
        std::vector<float> &F1);
    std::pair<uint32_t, float> predict(float const * const x) const;

private:
    std::shared_ptr<TreeNode> root;
};

class GBDT
{
public:
    GBDT(uint32_t const nr_tree) : trees(nr_tree), bias(0) {}
    void fit(Problem const &Tr, Problem const &Va);
    float predict(float const * const x) const;
    std::vector<uint32_t> get_indices(float const * const x) const;

private:
    std::vector<CART> trees;
    float bias;
};
