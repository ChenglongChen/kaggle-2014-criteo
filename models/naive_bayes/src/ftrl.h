#ifndef _FTRL_H_
#define _FTRL_H_

#include <string>
#include <vector>

#include "learner.h"

class FTRL : public Learner
{
public:
    FTRL(void) : pos(0), neg(0), likelihood(0), evidence(0) {};
    void load();
    void save();
    void update(int const y, std::vector<uint> const &idx, 
        std::vector<double> const &val);
    long long pos, neg;
    std::vector<long long> likelihood, evidence;
};

#endif // _FTRL_H_
