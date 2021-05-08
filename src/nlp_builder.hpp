#ifndef _NLP_BUILDER_HPP_
#define _NLP_BUILDER_HPP_

#include "c_nlp.hpp"
#include "nlp_base.hpp"
#include "null_nlp.hpp"
#include "py_nlp.hpp"

using FCallable = std::variant<ipyopt::py::F, ipyopt::c::F>;
using GradFCallable = std::variant<ipyopt::py::GradF, ipyopt::c::GradF>;
using GCallable = std::variant<ipyopt::py::G, ipyopt::c::G>;
using JacGCallable = std::variant<ipyopt::py::JacG, ipyopt::c::JacG>;
using HCallable = std::variant<ipyopt::null::H, ipyopt::py::H, ipyopt::c::H>;
using IntermediateCallbackCallable =
    std::variant<ipyopt::null::IntermediateCallback,
                 ipyopt::py::IntermediateCallback,
                 ipyopt::c::IntermediateCallback>;

inline bool is_null(const HCallable &h) {
  return std::holds_alternative<ipyopt::null::H>(h);
}

inline std::tuple<Ipopt::TNLP *, NlpData *>
build_nlp(FCallable &eval_f, GradFCallable &eval_grad_f, GCallable &eval_g,
          JacGCallable &eval_jac_g, SparsityIndices &&sparsity_indices_jac_g,
          HCallable &eval_h, SparsityIndices &&sparsity_indices_h,
          IntermediateCallbackCallable &intermediate_callback,
          std::vector<Ipopt::Number> &&x_l, std::vector<Ipopt::Number> &&x_u,
          std::vector<Ipopt::Number> &&g_l, std::vector<Ipopt::Number> &&g_u) {
  auto build = [&](auto &&eval_f, auto &&eval_grad_f, auto &&eval_g,
                   auto &&eval_jac_g, auto &&eval_h,
                   auto &&intermediate_callback) {
    auto nlp_base = new NlpBase{std::move(eval_f),
                                std::move(eval_grad_f),
                                std::move(eval_g),
                                std::move(eval_jac_g),
                                std::move(sparsity_indices_jac_g),
                                std::move(eval_h),
                                std::move(sparsity_indices_h),
                                std::move(intermediate_callback),
                                std::move(x_l),
                                std::move(x_u),
                                std::move(g_l),
                                std::move(g_u)};
    return std::tuple<Ipopt::TNLP *, NlpData *>{nlp_base, nlp_base};
  };
  return std::visit(build, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h,
                    intermediate_callback);
}

#endif
