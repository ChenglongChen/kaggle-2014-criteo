#ifndef _LEANER_H_
#define _LEANER_H_

#include <string>
#include <vector>

struct Parameter { };

class Learner
{
public:
    virtual void load() = 0;
    virtual void save() = 0;
    virtual void update(int const y, std::vector<uint> const &idx, 
        std::vector<double> const &val) = 0;
    void convert(std::string const &path);
    std::vector<double> const &get_W();
protected:
    std::vector<double> W;
    static const std::string model_path;
};

#endif // _LEANER_H_
