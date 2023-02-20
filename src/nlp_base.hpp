#ifndef _NLP_BASE_H_
#define _NLP_BASE_H_

#include "IpIpoptApplication.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptData.hpp"
#include "IpOrigIpoptNLP.hpp"
#include "IpTNLP.hpp"

#include <functional>
#include <variant>
#include <vector>

using SparsityIndices = std::tuple<std::vector<int>, std::vector<int>>;
using IpoptOptionValue = std::variant<int, double, char *>;

void arr_copy(const double *src, double *dest, std::size_t n);
void copy_sparsity(const SparsityIndices &sparsity_indices, Ipopt::Index *iRow,
                   Ipopt::Index *jCol);

/**
 * @brief Struct holding stats about an optimization run.
 */
struct NlpStats {
  Ipopt::Number n_eval_f;
  Ipopt::Number n_eval_grad_f;
  Ipopt::Number n_eval_g_eq;
  Ipopt::Number n_eval_jac_g_eq;
  Ipopt::Number n_eval_g_ineq;
  Ipopt::Number n_eval_jac_g_ineq;
  Ipopt::Number n_eval_h;
  Ipopt::Number n_iter;
  inline NlpStats()
      : n_eval_f{-1}, n_eval_grad_f{-1}, n_eval_g_eq{-1}, n_eval_jac_g_eq{-1},
        n_eval_g_ineq{-1}, n_eval_jac_g_ineq{-1}, n_eval_h{-1}, n_iter{-1} {}
};

/**
 * @brief Data only part of an IPopt NLP.
 */
class NlpData {
protected:
  Ipopt::Index _n, _m;
  Ipopt::Number *_out_x;
  Ipopt::Number *_out_z_L;
  Ipopt::Number *_out_z_U;
  Ipopt::Number *_out_g;
  Ipopt::Number *_out_lambda;
  Ipopt::Number _out_obj_value;

public:
  const Ipopt::Index &n, &m;
  const Ipopt::Number &out_obj_value;
  NlpStats out_stats;

  std::vector<Ipopt::Number> _x_scaling, _g_scaling;
  Ipopt::Number _obj_scaling;
  NlpData(Ipopt::Index n, Ipopt::Index m);
  void set_initial_values(double *x0, double *mult_g, double *mult_x_L,
                          double *mult_x_U);
};

/**
 * @brief Ipopt NLP templated interface, exposing generic callables.
 */
template <class F, class GradF, class G, class JacG, class H,
          class IntermediateCb>
class NlpBase : public NlpData, public Ipopt::TNLP {
protected:
  std::vector<Ipopt::Number> _x_l, _x_u, _g_l, _g_u;
  SparsityIndices _sparsity_indices_jac_g, _sparsity_indices_h;
  std::size_t _nnz_jac_g, _nnz_h;
  F _eval_f;
  GradF _eval_grad_f;
  G _eval_g;
  JacG _eval_jac_g;
  H _eval_h;
  IntermediateCb _intermediate_callback;

public:
  NlpBase(F &&eval_f, GradF &&eval_grad_f, G &&eval_g, JacG &&eval_jac_g,
          SparsityIndices &&sparsity_indices_jac_g, H &&eval_h,
          SparsityIndices &&sparsity_indices_h,
          IntermediateCb &&intermediate_callback,
          std::vector<Ipopt::Number> &&x_l, std::vector<Ipopt::Number> &&x_u,
          std::vector<Ipopt::Number> &&g_l, std::vector<Ipopt::Number> &&g_u)
      : NlpData{(Ipopt::Index)x_l.size(), (Ipopt::Index)g_l.size()},
        _x_l{std::move(x_l)}, _x_u{std::move(x_u)}, _g_l{std::move(g_l)},
        _g_u{std::move(g_u)}, _sparsity_indices_jac_g{std::move(
                                  sparsity_indices_jac_g)},
        _sparsity_indices_h{std::move(sparsity_indices_h)}, _eval_f{std::move(
                                                                eval_f)},
        _eval_grad_f{std::move(eval_grad_f)}, _eval_g{std::move(eval_g)},
        _eval_jac_g{std::move(eval_jac_g)}, _eval_h{std::move(eval_h)},
        _intermediate_callback{std::move(intermediate_callback)} {
    _nnz_jac_g = std::get<0>(_sparsity_indices_jac_g).size();
    _nnz_h = std::get<0>(_sparsity_indices_h).size();
  }

  virtual ~NlpBase() {}

  /**@name Overloaded from TNLP */
  //@{
  /** Return the objective value */
  virtual bool eval_f(Ipopt::Index n, const Ipopt::Number *x,
                      bool, //new_x,
                      Ipopt::Number &obj_value) {
    static_assert(std::is_constructible<
                      std::function<bool(Ipopt::Index n, const Ipopt::Number *x,
                                         Ipopt::Number &obj_value)>,
                      F>::value,
                  "Invalid type of F.");
    return _eval_f(n, x, obj_value);
  }
  /** Return the gradient of the objective */
  virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number *x,
                           bool, //new_x,
                           Ipopt::Number *grad_f) {
    static_assert(std::is_constructible<
                      std::function<bool(Ipopt::Index, const Ipopt::Number *,
                                         Ipopt::Number *)>,
                      GradF>::value,
                  "Invalid type of GradF.");
    return _eval_grad_f(n, x, grad_f);
  }

  /** Return the constraint residuals */
  virtual bool eval_g(Ipopt::Index n, const Ipopt::Number *x,
                      bool, //new_x,
                      Ipopt::Index m, Ipopt::Number *g) {
    static_assert(std::is_constructible<
                      std::function<bool(Ipopt::Index, const Ipopt::Number *,
                                         Ipopt::Index, Ipopt::Number *)>,
                      G>::value,
                  "Invalid type of G.");
    return _eval_g(n, x, m, g);
  }

  /** Return:
   *   1) The structure of the jacobian (if "values" is nullptr)
   *   2) The values of the jacobian (if "values" is not nullptr)
   */
  virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number *x,
                          bool, //   new_x,
                          Ipopt::Index m, Ipopt::Index nele_jac,
                          Ipopt::Index *iRow, Ipopt::Index *jCol,
                          Ipopt::Number *values) {
    static_assert(
        std::is_constructible<
            std::function<bool(Ipopt::Index, const Ipopt::Number *,
                               Ipopt::Index, Ipopt::Index, Ipopt::Number *)>,
            JacG>::value,
        "Invalid type of JacG.");
    if (values == nullptr) {
      // return the structure of the Jacobian
      copy_sparsity(_sparsity_indices_jac_g, iRow, jCol);
      return true;
    }
    // return the values.
    return _eval_jac_g(n, x, m, nele_jac, values);
  }

  /** Return:
   *   1) The structure of the hessian of the lagrangian (if "values" is nullptr)
   *   2) The values of the hessian of the lagrangian (if "values" is not nullptr)
   */
  virtual bool eval_h(Ipopt::Index n, const Ipopt::Number *x,
                      bool, // new_x
                      Ipopt::Number obj_factor, Ipopt::Index m,
                      const Ipopt::Number *lambda,
                      bool, // new_lambda
                      Ipopt::Index nele_hess, Ipopt::Index *iRow,
                      Ipopt::Index *jCol, Ipopt::Number *values) {
    static_assert(std::is_constructible<
                      std::function<bool(Ipopt::Index, const Ipopt::Number *,
                                         Ipopt::Number, Ipopt::Index,
                                         const Ipopt::Number *, Ipopt::Index,
                                         Ipopt::Number *)>,
                      H>::value,
                  "Invalid type of H.");
    if (values == nullptr) { // return the structure
      copy_sparsity(_sparsity_indices_h, iRow, jCol);
      return true;
    }
    // return the values.
    return _eval_h(n, x, obj_factor, m, lambda, nele_hess, values);
  }

  virtual bool
  intermediate_callback(Ipopt::AlgorithmMode mode, Ipopt::Index iter,
                        Ipopt::Number obj_value, Ipopt::Number inf_pr,
                        Ipopt::Number inf_du, Ipopt::Number mu,
                        Ipopt::Number d_norm, Ipopt::Number regularization_size,
                        Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
                        Ipopt::Index ls_trials, const Ipopt::IpoptData *ip_data,
                        Ipopt::IpoptCalculatedQuantities *ip_cq) {
    static_assert(
        std::is_constructible<
            std::function<bool(
                Ipopt::AlgorithmMode, Ipopt::Index, Ipopt::Number,
                Ipopt::Number, Ipopt::Number, Ipopt::Number, Ipopt::Number,
                Ipopt::Number, Ipopt::Number, Ipopt::Number, Ipopt::Index,
                const Ipopt::IpoptData *, Ipopt::IpoptCalculatedQuantities *)>,
            IntermediateCb>::value,
        "Invalid type of IntermediateCb.");
    return _intermediate_callback(mode, iter, obj_value, inf_pr, inf_du, mu,
                                  d_norm, regularization_size, alpha_du,
                                  alpha_pr, ls_trials, ip_data, ip_cq);
  }

  /** Return some info about the NLP */
  virtual bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &m,
                            Ipopt::Index &nnz_jac_g, Ipopt::Index &nnz_h_lag,
                            IndexStyleEnum &index_style) {
    n = _n;
    m = _m;
    nnz_jac_g = _nnz_jac_g;
    nnz_h_lag = _nnz_h;

    index_style = TNLP::C_STYLE; // use C style indexing (0-based)

    return true;
  }

  /** Return the bounds for the problem */
  virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l,
                               Ipopt::Number *x_u, Ipopt::Index m,
                               Ipopt::Number *g_l, Ipopt::Number *g_u) {
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    arr_copy(_x_l.data(), x_l, n);
    arr_copy(_x_u.data(), x_u, n);
    arr_copy(_g_l.data(), g_l, m);
    arr_copy(_g_u.data(), g_u, m);

    return true;
  }

  /**
   * @brief Return the starting point for the algorithm
   */
  virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number *x,
                                  bool init_z, Ipopt::Number *z_L,
                                  Ipopt::Number *z_U, Ipopt::Index m,
                                  bool init_lambda, Ipopt::Number *lambda) {
    if (init_x)
      arr_copy(_out_x, x, n);
    if (init_z) {
      if (_out_z_L != nullptr)
        arr_copy(_out_z_L, z_L, n);
      if (_out_z_U != nullptr)
        arr_copy(_out_z_U, z_U, n);
    }
    if (init_lambda && _out_lambda != nullptr)
      arr_copy(_out_lambda, lambda, m);

    return (!init_z || (_out_z_L != nullptr && _out_z_U != nullptr)) &&
           (!init_lambda || _out_lambda != nullptr);
  }

  /** 
   * @brief This method is called when the algorithm is complete so the TNLP can store/write the solution
   */
  virtual void finalize_solution(Ipopt::SolverReturn, //status,
                                 Ipopt::Index n, const Ipopt::Number *x,
                                 const Ipopt::Number *z_L,
                                 const Ipopt::Number *z_U, Ipopt::Index m,
                                 const Ipopt::Number *g,
                                 const Ipopt::Number *lambda,
                                 Ipopt::Number obj_value,
                                 const Ipopt::IpoptData *ip_data,
                                 Ipopt::IpoptCalculatedQuantities *ip_cq) {
    if (_out_x != nullptr)
      arr_copy(x, _out_x, n);
    if (_out_z_L != nullptr)
      arr_copy(z_L, _out_z_L, n);
    if (_out_z_U != nullptr)
      arr_copy(z_U, _out_z_U, n);
    if (_out_g != nullptr)
      arr_copy(g, _out_g, m);
    if (_out_lambda != nullptr)
      arr_copy(lambda, _out_lambda, m);
    _out_obj_value = obj_value;
    if (ip_cq != NULL) {
      auto orignlp =
          dynamic_cast<Ipopt::OrigIpoptNLP *>(GetRawPtr(ip_cq->GetIpoptNLP()));
      out_stats.n_eval_f = orignlp->f_evals();
      out_stats.n_eval_grad_f = orignlp->grad_f_evals();
      out_stats.n_eval_g_eq = orignlp->c_evals();
      out_stats.n_eval_jac_g_eq = orignlp->jac_c_evals();
      out_stats.n_eval_g_ineq = orignlp->d_evals();
      out_stats.n_eval_jac_g_ineq = orignlp->jac_d_evals();
      out_stats.n_eval_h = orignlp->h_evals();
      out_stats.n_iter = ip_data->iter_count();
    }
  }

  //@}

  virtual bool get_scaling_parameters(Ipopt::Number &obj_scaling,
                                      bool &use_x_scaling, Ipopt::Index n,
                                      Ipopt::Number *x_scaling,
                                      bool &use_g_scaling, Ipopt::Index m,
                                      Ipopt::Number *g_scaling) {
    use_x_scaling = !_x_scaling.empty();
    if (use_x_scaling)
      arr_copy(_x_scaling.data(), x_scaling, n);
    use_g_scaling = !_g_scaling.empty();
    if (use_g_scaling)
      arr_copy(_g_scaling.data(), g_scaling, m);
    obj_scaling = _obj_scaling;
    return true;
  }

private:
  /**@name Methods to block default (automatically generated) compiler methods.
   *
   * (See Scott Meyers book, "Effective C++")
   */
  //@{
  NlpBase(const NlpBase &);
  NlpBase &operator=(const NlpBase &);
  //@}
};

class NlpBundle {
  Ipopt::SmartPtr<Ipopt::TNLP> _nlp;
  Ipopt::SmartPtr<Ipopt::IpoptApplication> _app;

public:
  NlpBundle();
  operator bool();
  inline void take_nlp(Ipopt::TNLP *nlp) { _nlp = nlp; }
  int optimize();
  bool set_option(const char *key, const IpoptOptionValue &value);
  void without_hess();
};

struct IpoptOption {
  enum Type { Number, Integer, String, Unknown };
  std::string name;
  Type type;
  std::string description_short;
  std::string description_long;
  std::string category;
};

std::vector<IpoptOption> get_ipopt_options();
#endif
