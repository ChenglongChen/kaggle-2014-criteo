#ifndef _EVAL_H
#define _EVAL_H

#include <vector>

typedef std::vector<double> dvec_t;
typedef std::vector<int>    ivec_t;

double calc_auc(const dvec_t& dec_values, const ivec_t& ty);
double calc_accuracy(const dvec_t& dec_values, const ivec_t& ty);
double calc_ctr_mae(const dvec_t& dec_values, const ivec_t& ty, const double rate);
double calc_log_loss(const dvec_t& dec_values, const ivec_t& ty, const double rate);

#endif
