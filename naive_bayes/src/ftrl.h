#ifndef _FTRL_H_
#define _FTRL_H_

#include <string>
#include <vector>

#include "learner.h"

struct FTRLParameter : public Parameter
{
    FTRLParameter() : alpha(-1), beta(-1), lambda1(-1), lambda2(-1) {} ;
    FTRLParameter(
        double const alpha, 
        double const beta, 
        double const lambda1, 
        double const lambda2) 
            : alpha(alpha), beta(beta), lambda1(lambda1), lambda2(lambda2) {} ;
    double alpha, beta, lambda1, lambda2;
};

class FTRL : public Learner
{
public:
    FTRL() : param(), Z(0), N(0) {};
    FTRL(FTRLParameter const &param) : param(param), Z(0), N(0) {};
    void load();
    void save();
    void update(int const y, std::vector<uint> const &idx, 
        std::vector<double> const &val);
private:
    FTRLParameter param;
    std::vector<double> Z, N;
};

#endif // _FTRL_H_
