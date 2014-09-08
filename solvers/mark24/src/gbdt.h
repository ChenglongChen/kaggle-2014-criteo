#include <vector>

#include "common.h"

struct TreeNode
{
    TreeNode() : is_leaf(true) {}
    std::vector<size_t> II;
    size_t feature;
    float threshold, dec_val;
    bool is_leaf;
    std::shared_ptr<TreeNode> left, right;

    void fit(std::vector<std::vector<Node>> const &X, std::vector<float> const &R)
};

class CART 
{
    void fit(Problem const &problem);
    void save(std::string const &path);
    void load(std::string const &path);

private:
    std::shared_ptr<TreeNode> root;
};

class GBDT
{
    GBDT(size_t const nr_tree) : trees(nr_tree) {}
    void fit(Problem const &problem);
    void save(std::string const &path);
    void load(std::string const &path);

private:
    std::vector<CART> trees;
};
