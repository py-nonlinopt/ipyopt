#include "py_nlp.hpp"
#include "py_helpers.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ipyopt_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

namespace {

PyObject *wrap_array(unsigned int n, Ipopt::Number *x) {
  npy_intp dims[] = {static_cast<npy_intp>(n)};
  return PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (char *)x);
}

PyObject *wrap_const_array(unsigned int n, const Ipopt::Number *x) {
  npy_intp dims[] = {static_cast<npy_intp>(n)};
  return PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (char *)x);
}

bool consume(PyObject *obj) {
  if (obj == nullptr)
    return false;
  Py_XDECREF(obj);
  return true;
}

} // namespace

namespace ipyopt {
namespace py {

PyObjSlot::PyObjSlot(PyObject *obj) : _obj{obj} {}

bool F::operator()(Ipopt::Index n, const Ipopt::Number *x,
                   Ipopt::Number &obj_value) {
  PyObject *py_x_arr = wrap_const_array(n, x);
  if (!py_x_arr)
    return false;
  PyObject *return_value = py_call(_obj, py_x_arr);
  if (return_value == nullptr)
    return false;
  obj_value = PyFloat_AsDouble(return_value);
  Py_XDECREF(return_value);
  if (PyErr_Occurred()) {
    PyErr_Format(PyExc_RuntimeError,
                 "Python function eval_f returns non-PyFloat");
    return false;
  }
  return true;
}
bool GradF::operator()(Ipopt::Index n, const Ipopt::Number *x,
                       Ipopt::Number *grad_f) {
  return consume(py_call(_obj, wrap_const_array(n, x), wrap_array(n, grad_f)));
}

bool G::operator()(Ipopt::Index n, const Ipopt::Number *x, Ipopt::Index m,
                   Ipopt::Number *g) {
  return consume(py_call(_obj, wrap_const_array(n, x), wrap_array(m, g)));
}

bool JacG::operator()(Ipopt::Index n, const Ipopt::Number *x,
                      Ipopt::Index /*m*/, Ipopt::Index nele_jac,
                      Ipopt::Number *values) {
  // return the values of the Jacobian of the constraints
  return consume(
      py_call(_obj, wrap_const_array(n, x), wrap_array(nele_jac, values)));
}

bool H::operator()(Ipopt::Index n, const Ipopt::Number *x,
                   Ipopt::Number obj_factor, Ipopt::Index m,
                   const Ipopt::Number *lambda, Ipopt::Index nele_hess,
                   Ipopt::Number *values) {
  return consume(
      py_call(_obj, wrap_const_array(n, x), wrap_const_array(m, lambda),
              PyFloat_FromDouble(obj_factor), wrap_array(nele_hess, values)));
}

bool IntermediateCallback::operator()(
    Ipopt::AlgorithmMode mode, Ipopt::Index iter, Ipopt::Number obj_value,
    Ipopt::Number inf_pr, Ipopt::Number inf_du, Ipopt::Number mu,
    Ipopt::Number d_norm, Ipopt::Number regularization_size,
    Ipopt::Number alpha_du, Ipopt::Number alpha_pr, Ipopt::Index ls_trials,
    const Ipopt::IpoptData * /*ip_data*/,
    Ipopt::IpoptCalculatedQuantities * /*ip_cq*/) {
  auto *result = py_call(
      _obj, PyLong_FromLong(mode), PyLong_FromLong(iter),
      PyFloat_FromDouble(obj_value), PyFloat_FromDouble(inf_pr),
      PyFloat_FromDouble(inf_du), PyFloat_FromDouble(mu),
      PyFloat_FromDouble(d_norm), PyFloat_FromDouble(regularization_size),
      PyFloat_FromDouble(alpha_du), PyFloat_FromDouble(alpha_pr),
      PyLong_FromLong(ls_trials));
  if (result == nullptr) {
    PyErr_Print();
    return false;
  }
  bool result_as_bool = PyLong_AsLong(result);
  Py_DECREF(result);

  if (PyErr_Occurred()) {
    PyErr_Format(PyExc_RuntimeError,
                 "Python function intermediate_callback returned non bool");
    return false;
  }
  return result_as_bool;
}
} // namespace py
} // namespace ipyopt
