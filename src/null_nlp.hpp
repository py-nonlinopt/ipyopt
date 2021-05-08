#ifndef _NULL_NLP_H_
#define _NULL_NLP_H_

#include "IpTNLP.hpp"

namespace ipyopt {
namespace null {
struct H {
  inline bool operator()(Ipopt::Index, const Ipopt::Number *, Ipopt::Number,
                         Ipopt::Index, const Ipopt::Number *, Ipopt::Index,
                         Ipopt::Number *) {
    return true;
  }
};
struct IntermediateCallback {
  inline bool operator()(Ipopt::AlgorithmMode, Ipopt::Index, Ipopt::Number,
                         Ipopt::Number, Ipopt::Number, Ipopt::Number,
                         Ipopt::Number, Ipopt::Number, Ipopt::Number,
                         Ipopt::Number, Ipopt::Index,
                         const Ipopt::IpoptData * /*ip_data*/,
                         Ipopt::IpoptCalculatedQuantities * /*ip_cq*/) {
    return true;
  }
};

} // namespace null
} // namespace ipyopt

#endif
