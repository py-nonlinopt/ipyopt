#ifndef _C_NLP_H_
#define _C_NLP_H_

#include "nlp_base.hpp"

namespace ipyopt {
namespace c {
template <typename... Args> struct LowLevelCallable {
  using FunctionType = bool(Args..., void *);
  FunctionType *function;
  void *user_data;
  bool operator()(Args... args) { return function(args..., user_data); }
  LowLevelCallable() : function{nullptr}, user_data{nullptr} {}
};

struct F : public LowLevelCallable<Ipopt::Index, const Ipopt::Number *,
                                   Ipopt::Number *> {
  inline bool operator()(Ipopt::Index n, const Ipopt::Number *x,
                         Ipopt::Number &obj_value) {
    return function(n, x, &obj_value, user_data);
  }
};

using GradF =
    LowLevelCallable<Ipopt::Index, const Ipopt::Number *, Ipopt::Number *>;

using G = LowLevelCallable<Ipopt::Index, const Ipopt::Number *, Ipopt::Index,
                           Ipopt::Number *>;

using JacG = LowLevelCallable<Ipopt::Index, const Ipopt::Number *, Ipopt::Index,
                              Ipopt::Index, Ipopt::Number *>;

using H = LowLevelCallable<Ipopt::Index, const Ipopt::Number *, Ipopt::Number,
                           Ipopt::Index, const Ipopt::Number *, Ipopt::Index,
                           Ipopt::Number *>;

using IntermediateCallback =
    LowLevelCallable<Ipopt::AlgorithmMode, Ipopt::Index, Ipopt::Number,
                     Ipopt::Number, Ipopt::Number, Ipopt::Number, Ipopt::Number,
                     Ipopt::Number, Ipopt::Number, Ipopt::Number, Ipopt::Index,
                     const Ipopt::IpoptData *,
                     Ipopt::IpoptCalculatedQuantities *>;
} // namespace c
} // namespace ipyopt
#endif
