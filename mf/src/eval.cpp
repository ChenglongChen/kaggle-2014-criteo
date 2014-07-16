#include <iostream>
#include <vector>
#include <algorithm>
#include <errno.h>
#include <cstring>
#include "eval.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class Comp{
  const double *dec_val;
  public:
  Comp(const double *ptr): dec_val(ptr){}
  bool operator()(int i, int j) const{
    return dec_val[i] > dec_val[j];
  }
};

double calc_auc(const dvec_t& dec_values, const ivec_t& ty){
  double roc  = 0;
  size_t size = dec_values.size();
  size_t i;
  std::vector<size_t> indices(size);

  for(i = 0; i < size; ++i) indices[i] = i;

  std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));

  int tp = 0,fp = 0;
  for(i = 0; i < size; i++) {
    if (ty[indices[i]] == 1) tp++;
    else if (ty[indices[i]] == -1) {
      roc += tp;
      fp++;
    }
  }

  if (tp == 0 || fp == 0)
  {
    fprintf(stderr, "warning: Too few postive true labels or negative true labels\n");
    roc = 0;
  }
  else
    roc = roc / tp / fp;

  return roc;
}

double calc_accuracy(const dvec_t& dec_values, const ivec_t& ty){
  int    correct = 0;
  int    total   = (int) ty.size();
  size_t i;

  for(i = 0; i < ty.size(); ++i)
    if (ty[i] == (dec_values[i] >= 0? 1: -1)) ++correct;

  return (double) correct / total;
}

double calc_ctr_mae(const dvec_t& dec_values, const ivec_t& ty, const double rate){
  double click_real=0, click_pred=0;

  double const log_rate = log(rate);

  for(size_t i = 0; i < ty.size(); ++i) {
    //double const p_pos = 1/(1+exp(-dec_values[i]));
    //click_pred += (p_pos*rate)/(1+(rate-1)*p_pos);
    click_pred += 1/(1+exp(-(dec_values[i]+log_rate)));
    if (ty[i] == 1) click_real += 1;
  }

  if(click_real == 0)
    return -1;
  else   
    return (click_pred-click_real)/click_real;
}

inline double lr(double const wTx)
{
    return 1/(1+exp(-wTx));
}

double 
calc_log_loss(const dvec_t& dec_values, const ivec_t& ty, const double rate)
{
  double const log_rate = log(rate);

  double loss = 0;
  for(size_t i = 0; i < ty.size(); ++i) 
  {
    double prob = lr(dec_values[i]+log_rate);
    if(prob == 1 && ty[i] != 1)
        prob = 0.999;
    else if(prob == 0 && ty[i] == 1)
    {
        prob = 0.001;
    }
    double const loss1 = (ty[i]==1)? log(prob) : log(1-prob);
    loss += loss1;
  }
  return -loss/static_cast<double>(ty.size());
}
