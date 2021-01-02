#ifndef _PY_NLP_H_
#define _PY_NLP_H_

#include "Python.h"
#include "nlp_base.hpp"

namespace ipyopt {
namespace py {
class PyObjSlot {
protected:
  PyObject *_obj;

public:
  PyObjSlot(PyObject *obj = nullptr);
};

struct F : public PyObjSlot {
  bool operator()(Ipopt::Index n, const Ipopt::Number *x,
                  Ipopt::Number &obj_value);
};

struct GradF : public PyObjSlot {
  bool operator()(Ipopt::Index n, const Ipopt::Number *x,
                  Ipopt::Number *grad_f);
};

struct G : public PyObjSlot {
  bool operator()(Ipopt::Index n, const Ipopt::Number *x, Ipopt::Index m,
                  Ipopt::Number *g);
};

struct JacG : public PyObjSlot {
  bool operator()(Ipopt::Index n, const Ipopt::Number *x, Ipopt::Index m,
                  Ipopt::Index nele_jac, Ipopt::Number *values);
};

struct H : public PyObjSlot {
  bool operator()(Ipopt::Index n, const Ipopt::Number *x,
                  Ipopt::Number obj_factor, Ipopt::Index m,
                  const Ipopt::Number *lambda, Ipopt::Index nele_hess,
                  Ipopt::Number *values);
};

struct IntermediateCallback : public PyObjSlot {
  bool operator()(Ipopt::AlgorithmMode mode, Ipopt::Index iter,
                  Ipopt::Number obj_value, Ipopt::Number inf_pr,
                  Ipopt::Number inf_du, Ipopt::Number mu, Ipopt::Number d_norm,
                  Ipopt::Number regularization_size, Ipopt::Number alpha_du,
                  Ipopt::Number alpha_pr, Ipopt::Index ls_trials,
                  const Ipopt::IpoptData *ip_data,
                  Ipopt::IpoptCalculatedQuantities *ip_cq);
};
} // namespace py
} // namespace ipyopt
#endif
