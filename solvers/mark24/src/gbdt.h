#include <vector>

#include "common.h"

struct TreeNode
{
    std::vector<size_t> I;
    size_t feature;
    float threshold;
    std::shared_ptr<TreeNode> left, right;

    void split(Problem const &problem, float const threshold);
};

class CART 
{
    void fit(Problem const &problem, std::vector<float> const &R);
    void save(std::string const &path);
    void load(std::string const &path);

private:
    std::shared_ptr<TreeNode> root;
};

class GBDT
{
    void fit(Problem const &problem);
    void save(std::string const &path);
    void load(std::string const &path);

private:
    std::vector<CART> trees;
};
