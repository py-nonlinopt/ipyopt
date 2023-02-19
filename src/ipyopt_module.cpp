#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ipyopt_ARRAY_API
#include "numpy/arrayobject.h"

#include <optional>

#include "nlp_builder.hpp"
#include "py_helpers.hpp"
#include "py_nlp.hpp"

static bool _PyDict_Check(const PyObject *obj) {
  return PyDict_Check(obj);
} // Macro -> function
static bool check_array_dim(PyArrayObject *arr, unsigned int dim,
                            const char *name) {
  if ((unsigned int)PyArray_NDIM(arr) != 1) {
    PyErr_Format(PyExc_ValueError,
                 "%s has wrong number of dimensions. Expected %d, got %d", name,
                 1, PyArray_NDIM(arr));
    return false;
  }
  if (PyArray_DIMS(arr)[0] == dim)
    return true;
  PyErr_Format(PyExc_ValueError,
               "%s has wrong shape. Expected (%d,), found (%d,)", name, dim,
               PyArray_DIMS(arr)[0]);
  return false;
}
static double *optional_array_data(PyArrayObject *arr) {
  if (arr == nullptr)
    return nullptr;
  return (double *)PyArray_DATA(arr);
}
template <class T, bool ALLOW_0>
static bool check_vec_size(const std::vector<T> &vec, unsigned int size,
                           const char *name) {
  if constexpr (ALLOW_0) {
    if (vec.empty())
      return true;
  }
  if (vec.size() == size)
    return true;
  PyErr_Format(PyExc_ValueError, "%s has wrong size %d (expected: %d)", name,
               vec.size(), size);
  return false;
}

static bool parse_sparsity_indices(PyObject *obj, SparsityIndices &idx) {
  auto [py_rows, py_cols] = from_py_tuple<2>(obj, "Sparsity info");
  if (PyErr_Occurred())
    return false;
  const auto rows = from_py_sequence<int>(py_rows, "Sparsity info");
  if (PyErr_Occurred())
    return false;
  const auto cols = from_py_sequence<int>(py_cols, "Sparsity info");
  if (PyErr_Occurred())
    return false;

  if (rows.size() != cols.size()) {
    PyErr_Format(PyExc_ValueError,
                 "Sparsity info: length of row indices (%d) does not match "
                 "lenth of column indices (%d)",
                 rows.size(), cols.size());
    return false;
  }

  idx = std::make_tuple(rows, cols);
  return true;
}
static bool check_non_negative(int n, const char *name) {
  if (n >= 0)
    return true;
  PyErr_Format(PyExc_ValueError, "%s can't be negative", name);
  return false;
}
static bool check_kwargs(const PyObject *kwargs) {
  if (kwargs == nullptr || PyDict_Check(kwargs))
    return true;
  PyErr_Format(PyExc_RuntimeError,
               "C-API-Level Error: keywords are not of type dict");
  return false;
}

static bool check_optional(const PyObject *obj,
                           bool (*checker)(const PyObject *),
                           const char *obj_name, const char *type_name) {
  if (obj == nullptr || obj == Py_None || checker(obj))
    return true;
  PyErr_Format(PyExc_TypeError, "Wrong type for %s. Required: %s", obj_name,
               type_name);
  return false;
}
static bool check_no_args(const char *f_name, PyObject *args) {
  if (args == nullptr)
    return true;
  if (!PyTuple_Check(args)) {
    PyErr_Format(PyExc_RuntimeError, "Argument keywords is not a tuple");
    return false;
  }
  unsigned int n = PyTuple_Size(args);
  if (n == 0)
    return true;
  PyErr_Format(PyExc_TypeError,
               "%s() takes 0 positional arguments but %d %s given", f_name, n,
               n == 1 ? "was" : "were");
  return false;
}

template <const char *ArgName, bool ALLOW_0, typename T>
static int parse_vec(PyObject *obj, void *out) {
  std::vector<T> &vec = *(std::vector<T> *)out;
  if constexpr (ALLOW_0) {
    if (obj == Py_None || obj == nullptr) {
      vec.clear();
      return 1;
    }
  }
  if (!PyArray_Check(obj)) {
    PyErr_Format(PyExc_TypeError,
                 "%s() argument '%s' must be numpy.ndarray, not %s", "%s",
                 ArgName, Py_TYPE(obj)->tp_name);
    return 0;
  }
  PyArrayObject *arr = (PyArrayObject *)obj;
  if ((unsigned int)PyArray_NDIM(arr) != 1) {
    PyErr_Format(
        PyExc_ValueError,
        "%s() argument '%s': numpy.ndarray dimension must be 1, not %d", "%s",
        ArgName, PyArray_NDIM(arr));
    return 0;
  }
  vec.resize(PyArray_SIZE(arr));
  for (unsigned int i = 0; i < vec.size(); i++) {
    vec[i] = ((T *)PyArray_DATA(arr))[i];
  }
  return 1;
}

template <const char *ArgName, class LLC>
static bool parse_py_capsule(PyObject *obj, LLC &llc) {
  const auto name = PyCapsule_GetName(obj);
  if (!PyCapsule_IsValid(obj, name)) {
    PyErr_Format(PyExc_ValueError,
                 "%s() argument %s: invalid PyCapsule with name '%s'", "%s",
                 ArgName, (name != nullptr) ? name : "");
    return false;
  }
  llc.function =
      (typename LLC::FunctionType *)(PyCapsule_GetPointer(obj, name));
  llc.user_data = PyCapsule_GetContext(obj);
  return true;
}

/**
 * Parse a `scipy.LowLevelCallable`_ into 2 void pointers.
 *
 * A `scipy.LowLevelCallable`_ is a sub class of a 3 tuple
 * tuple[PyCapsule, Union, Union]
 * The actual callback is held in slot 0.
 * This PyCapsule also holds the userdata as its context.
 *
 * See https://github.com/scipy/scipy/blob/master/scipy/_lib/_ccallback.py
 * and https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html
 */
template <const char *ArgName, class LLC>
static bool parse_scipy_low_level_callable(PyObject *obj, LLC &llc) {
  auto capsule = PyTuple_GET_ITEM(obj, 0);
  if (capsule == nullptr) {
    PyErr_Format(PyExc_SystemError, "%s() argument '%s': invalid tuple", "%s",
                 ArgName);
  }
  return parse_py_capsule<ArgName, LLC>(capsule, llc);
}

/// This is for python memory management (parse py object and keep a pointer to the original py object at the same time)
template <class T> struct WithOwnedPyObject {
  T callable;
  PyObject *owned = nullptr;
};

template <const char *ArgName, class Variant, class PyCallable, class CCallable>
static int parse_callable(PyObject *obj, void *out) {
  auto &callable = *(WithOwnedPyObject<Variant> *)out;
  callable.owned = obj;
  if (PyCapsule_CheckExact(obj)) {
    CCallable llc;
    if (!parse_py_capsule<ArgName, CCallable>(obj, llc))
      return 0;
    callable.callable = llc;
    return 1;
  }
  if (PyTuple_Check(obj) && PyTuple_Size(obj) == 3) {
    CCallable llc;
    if (!parse_scipy_low_level_callable<ArgName, CCallable>(obj, llc))
      return 0;
    callable.callable = llc;
    return 1;
  }

  if (PyCallable_Check(obj)) {
    callable.callable = PyCallable{obj};
    return 1;
  }
  PyErr_Format(PyExc_ValueError,
               "%s() argument '%s': must be Union[Callable, PyCapsule, "
               "scipy.LowLevelCallable], not %s",
               "%s", ArgName, Py_TYPE(obj)->tp_name);
  return 0;
}

using PyConverter = int(PyObject *, void *);
template <PyConverter converter>
static int parse_optional(PyObject *obj, void *out) {
  if (obj == nullptr || obj == Py_None)
    return 1;
  return converter(obj, out);
}

static std::optional<IpoptOptionValue> py_unpack(PyObject *obj) {
  if (PyLong_Check(obj))
    return (int)PyLong_AsLong(obj);
  if (PyFloat_Check(obj))
    return (double)PyFloat_AsDouble(obj);
  if (PyUnicode_Check(obj))
    return (char *)PyUnicode_AsUTF8(obj);
  return std::nullopt;
}
static bool set_options(NlpBundle &bundle, PyObject *dict) {
  PyObject *key, *val;
  Py_ssize_t pos = 0;
  if (dict == nullptr)
    return true;
  while (PyDict_Next(dict, &pos, &key, &val)) {
    const char *c_key = PyUnicode_AsUTF8(key);
    std::optional<IpoptOptionValue> value = py_unpack(val);
    if (!value.has_value()) {
      PyErr_Format(PyExc_TypeError,
                   "The value for option %s has unsupported type", c_key);
      return false;
    }
    if (!bundle.set_option(c_key, value.value())) {
      PyErr_Format(PyExc_ValueError, "Failed to set the Ipopt option '%s'",
                   c_key);
      return false;
    }
  }
  return true;
}

static void reformat_error(const char *f_name) {
  PyObject *ptype, *pvalue, *ptraceback;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);

  const char *pStrErrorMessage = PyUnicode_AsUTF8(pvalue);
  PyErr_Format(ptype, pStrErrorMessage, f_name);
}

/// Python memory management:
constexpr std::size_t N_MEMBER_SLOTS = 6;

template <typename... Obj>
void receive_members(PyObject *member_slots[N_MEMBER_SLOTS], Obj *...members) {
  static_assert(sizeof...(members) <= N_MEMBER_SLOTS);
  std::size_t i = 0;
  for (auto *obj : {members...}) {
    Py_XINCREF(obj);
    member_slots[i++] = obj;
  }
}

extern "C" {
typedef struct {
  PyObject_HEAD // ---
      NlpBundle *bundle;
  NlpData *nlp;
  // Python memory management:
  PyObject *member_slots[N_MEMBER_SLOTS];
} PyNlpApp;

static int py_ipopt_problem_clear(PyNlpApp *self) {
  for (std::size_t i = 0; i < N_MEMBER_SLOTS; i++)
    Py_CLEAR(self->member_slots[i]);
  return 0;
}
static void py_ipopt_problem_dealloc(PyNlpApp *self) {
  PyObject_GC_UnTrack(self);
  py_ipopt_problem_clear(self);
  if (self->bundle != nullptr) {
    delete self->bundle;
    self->bundle = nullptr;
  }
  Py_TYPE(self)->tp_free((PyObject *)self);
}
static int py_ipopt_problem_traverse(PyNlpApp *self, visitproc visit,
                                     void *arg) {
  for (std::size_t i = 0; i < N_MEMBER_SLOTS; i++)
    if (self->member_slots[i] != nullptr) {
      Py_VISIT(self->member_slots[i]);
    }
  return 0;
}

// Cannot pass string literals as template args. Therefor use static strings:
constexpr char arg_x_l[] = "x_l";
constexpr char arg_x_u[] = "x_u";
constexpr char arg_g_l[] = "g_l";
constexpr char arg_g_u[] = "g_u";
constexpr char arg_f[] = "eval_f";
constexpr char arg_grad_f[] = "eval_grad_f";
constexpr char arg_g[] = "eval_g";
constexpr char arg_jac_g[] = "eval_jac_g";
constexpr char arg_h[] = "eval_h";
constexpr char arg_intermediate_callback[] = "intermediate_callback";
constexpr char arg_x_scaling[] = "x_scaling";
constexpr char arg_g_scaling[] = "g_scaling";

static char IPYOPT_PROBLEM_DOC[] = R"mdoc(
Ipopt problem type in python

Problem(n: int, x_l: numpy.ndarray[numpy.float64], x_u: numpy.ndarray[numpy.float64], m: int, g_l: numpy.ndarray[numpy.float64], g_u: numpy.ndarray[numpy.float64], sparsity_indices_jac_g: tuple[Sequence[float], Sequence[float]], sparsity_indices_h: tuple[Sequence[float], Sequence[float]], eval_f: Union[Callable[[numpy.ndarray], float], PyCapsule, scipy.LowLevelCallable], eval_grad_f: Union[Callable[[numpy.ndarray, numpy.ndarray], Any], PyCapsule, scipy.LowLevelCallable], eval_g: Union[Callable[[numpy.ndarray, numpy.ndarray], Any], PyCapsule, scipy.LowLevelCallable], eval_jac_g: Union[Callable[[numpy.ndarray, numpy.ndarray], Any], PyCapsule, scipy.LowLevelCallable], eval_h: Optional[Union[Callable[[numpy.ndarray, numpy.ndarray, float, numpy.ndarray], Any], PyCapsule, scipy.LowLevelCallable]] = None, intermediate_callback: Optional[Union[Callable[[int, int, float, float, float, float, float, float, float, float, int], Any], PyCapsule, scipy.LowLevelCallable]] = None, obj_scaling: float = 1., x_scaling: Optional[numpy.ndarray[numpy.float64]] = None, g_scaling: Optional[numpy.ndarray[numpy.float64]] = None, ipopt_options: Optional[dict[str, Union[int, float, str]]] = None) -> Problem

Args:
    n: Number of variables (dimension of ``x``)
    x_l: Lower bound of ``x`` as bounded constraints
    x_u: Upper bound of ``x`` as bounded constraints
        both ``x_l``, ``x_u`` should be one 1-dim arrays with length ``n``

    m: Number of constraints
    g_l: Lower bound of constraints
    g_u: Upper bound of constraints
        both ``g_l``, ``g_u`` should be one dimension arrays with length ``m``
    sparsity_indices_jac_g: Positions of non-zero entries of ``jac_g`` in the form of a tuple of two sequences of the same length (first list are constraint/row indices, second column are variable/column indices)
    sparsity_indices_h: Positions of non-zero entries of ``hess``
    eval_f: Callback function to calculate objective value.
        Signature: ``eval_f(x: numpy.ndarray) -> float``. Also accepts a `PyCapsule`_ / `scipy.LowLevelCallable`_ object. In this case, the C function has signature::
            
            bool f(int n, double* x, double *obj_value, void *user_data)

    eval_grad_f: calculates gradient for objective function.
        Signature: ``eval_grad_f(x: numpy.ndarray, out: numpy.ndarray) -> Any``.
        The array ``out`` must be a 1-dim array matching the length of ``x``, i.e. ``n``.
        A possible return value will be ignored.
        Also accepts a `PyCapsule`_ / `scipy.LowLevelCallable`_ object. In this case, the C function has signature::

            bool grad_f(int n, double* x, double *out, void *user_data)

    eval_g: calculates the constraint values and return an array
        The constraints are defined by ::

            g_l <= g(x) <= g_u

        Signature: ``eval_g(x: numpy.ndarray, out: numpy.ndarray) -> Any``.
        The array ``out`` must be a 1-dim array of length ``m``.
        A possible return value will be ignored.
        Also accepts a `PyCapsule`_ / `scipy.LowLevelCallable`_ object. In this case, the C function has signature::

            bool g(int n, double* x, int m, double *out, void *user_data)

    eval_jac_g: calculates the Jacobi matrix.
        Signature: ``eval_jac_g(x: numpy.ndarray, out: numpy.ndarray) -> Any``. The array ``out`` must be a 1-dim array whose entries are the entries of the Jacobi matrix `jac_g` listed in ``sparsity_indices_jac_g`` (order matters).
        A possible return value will be ignored.
        Also accepts a `PyCapsule`_ / `scipy.LowLevelCallable`_ object. In this case, the C function has signature::

            bool jac_g(int n,
                       double* x,
                       int m,
                       int nele_jac,
                       double *out,
                       void *user_data)

    eval_h: calculates the Hessian of the Lagrangian ``L`` (optional).
        Signature::

            eval_h(x: numpy.ndarray, lagrange: numpy.ndarray,
                   obj_factor: float, out: numpy.ndarray) -> Any

        The array ``out`` must be a 1-dim array and contain the entries of the Hessian of the Lagrangian L::

            L = obj_factor * f + lagrange[i] * g[i] (sum over `i`),

        listed in ``sparsity_indices_hess`` for given ``obj_factor: float``
        and ``lagrange: numpy.ndarray`` of shape ``(m,)``.
        A possible return value will be ignored.
        If omitted, the parameter ``sparsity_indices_hess`` will be ignored and Ipopt will use approximated hessian
        which will make the convergence slower.
        Also accepts a `PyCapsule`_ / `scipy.LowLevelCallable`_ object. In this case, the C function has signature::

            bool h(int n,
                   double* x,
                   double obj_value,
                   int m,
                   double *lagrange,
                   int nele_hess,
                   double *out,
                   void *user_data)

    intermediate_callback: Intermediate Callback method for the user.
        This method is called once per iteration (during the convergence check), and can be used to obtain information about the optimization status while Ipopt solves the problem, and also to request a premature termination (see the Ipopt docs for more details).
        Signature::

            intermediate_callback(
              mode: int, iter: int, obj_value: float,
              inf_pr: float, inf_du: float, mu: float,
              d_norm: float, regularization_size: float,
              alpha_du: float, alpha_pr: float,
              ls_trials: int
            ) -> Any

        Also accepts a `PyCapsule`_ / `scipy.LowLevelCallable`_ object. In this case, the C function has signature::

            bool intermediate_callback(int algorithm_mode,
                                       int iter,
                                       double obj_value,
                                       double inf_pr,
                                       double inf_du,
                                       double mu,
                                       double d_norm,
                                       double regularization_size,
                                       double alpha_du,
                                       double alpha_pr,
                                       int ls_trails,
                                       const void *ip_data,
                                       void *ip_cq,
                                       void *userdata)

    obj_scaling: A scaling factor for the objective value (see ``set_problem_scaling``).
    x_scaling: Either ``None`` (no scaling) or a ``numpy.ndarray`` of length ``n``, scaling the ``x`` variables (see :func:`set_problem_scaling`).
    g_scaling: Either ``None`` (no scaling) or a ``numpy.ndarray`` of length ``m``, scaling the ``g`` variables (see :func:`set_problem_scaling`).
    ipopt_options: A dict of key value pairs, to be passed to Ipopt (use :func:`get_ipopt_options` to get a list of all options available)"

.. _`PyCapsule`: https://docs.python.org/3/c-api/capsule.html
.. _`scipy.LowLevelCallable`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html?highlight=lowlevelcallable#scipy.LowLevelCallable
)mdoc";
static PyObject *py_ipopt_problem_new(PyTypeObject *type, PyObject *args,
                                      PyObject *keywords) {
  auto *self = (PyNlpApp *)type->tp_alloc(type, 0);
  for (std::size_t i = 0; i < N_MEMBER_SLOTS; i++)
    self->member_slots[i] = nullptr;
  self->bundle = new NlpBundle{};
  if (!*self->bundle) {
    delete self->bundle;
    self->bundle = nullptr;
    type->tp_free(self);
    PyErr_SetString(PyExc_MemoryError, "Cannot create IpoptProblem instance");
    return nullptr;
  }

  Ipopt::Index n, m;
  Ipopt::Number obj_scaling;
  PyObject *py_sparsity_indices_jac_g = nullptr;
  PyObject *py_sparsity_indices_h = nullptr;
  PyObject *py_ipopt_options = nullptr;
  std::vector<double> x_scaling, g_scaling, x_l, x_u, g_l, g_u;
  WithOwnedPyObject<FCallable> py_eval_f;
  WithOwnedPyObject<GradFCallable> py_eval_grad_f;
  WithOwnedPyObject<GCallable> py_eval_g;
  WithOwnedPyObject<JacGCallable> py_eval_jac_g;
  WithOwnedPyObject<HCallable> py_eval_h;
  WithOwnedPyObject<IntermediateCallbackCallable> py_intermediate_callback;
  SparsityIndices sparsity_indices_jac_g, sparsity_indices_h;
  const char *arg_names[] = {"n",
                             "x_l",
                             "x_u",
                             "m",
                             "g_l",
                             "g_u",
                             "sparsity_indices_jac_g",
                             "sparsity_indices_h",
                             "eval_f",
                             "eval_grad_f",
                             "eval_g",
                             "eval_jac_g",
                             "eval_h",
                             "intermediate_callback",
                             "obj_scaling",
                             "x_scaling",
                             "g_scaling",
                             "ipopt_options",
                             nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, keywords,
          "iO&O&iO&O&OOO&O&O&O&|O&O&dO&O&O:%s", // function name will be substituted later
          const_cast<char **>(arg_names), &n,
          &parse_vec<arg_x_l, false, double>, &x_l,
          &parse_vec<arg_x_u, false, double>, &x_u, &m,
          &parse_vec<arg_g_l, false, double>, &g_l,
          &parse_vec<arg_g_u, false, double>, &g_u, &py_sparsity_indices_jac_g,
          &py_sparsity_indices_h,
          &parse_callable<arg_f, FCallable, ipyopt::py::F, ipyopt::c::F>,
          &py_eval_f,
          &parse_callable<arg_grad_f, GradFCallable, ipyopt::py::GradF,
                          ipyopt::c::GradF>,
          &py_eval_grad_f,
          &parse_callable<arg_g, GCallable, ipyopt::py::G, ipyopt::c::G>,
          &py_eval_g,
          &parse_callable<arg_jac_g, JacGCallable, ipyopt::py::JacG,
                          ipyopt::c::JacG>,
          &py_eval_jac_g,
          &parse_optional<
              parse_callable<arg_h, HCallable, ipyopt::py::H, ipyopt::c::H>>,
          &py_eval_h,
          &parse_optional<parse_callable<arg_intermediate_callback,
                                         IntermediateCallbackCallable,
                                         ipyopt::py::IntermediateCallback,
                                         ipyopt::c::IntermediateCallback>>,
          &py_intermediate_callback, &obj_scaling,
          &parse_vec<arg_x_scaling, true, double>, &x_scaling,
          &parse_vec<arg_g_scaling, true, double>, &g_scaling,
          &py_ipopt_options) ||
      !parse_sparsity_indices(py_sparsity_indices_jac_g,
                              sparsity_indices_jac_g) ||
      !check_non_negative(n, "n") || !check_non_negative(m, "m") ||
      !check_vec_size<double, false>(x_l, n, "%s() argument x_L") ||
      !check_vec_size<double, false>(x_u, n, "%s() argument x_U") ||
      !check_vec_size<double, false>(g_l, m, "%s() argument g_L") ||
      !check_vec_size<double, false>(g_u, m, "%s() argument g_U") ||
      !(is_null(py_eval_h.callable) ||
        parse_sparsity_indices(py_sparsity_indices_h, sparsity_indices_h)) ||
      !check_optional(py_ipopt_options, _PyDict_Check, "ipopt_options",
                      "Optional[dict]]") ||
      !check_vec_size<double, true>(x_scaling, n, "%s() argument x_scaling") ||
      !check_vec_size<double, true>(g_scaling, m, "%s() argument g_scaling") ||
      !set_options(*self->bundle, py_ipopt_options)) {
    if (self->bundle != nullptr) {
      delete self->bundle;
      self->bundle = nullptr;
    }
    Py_CLEAR(self);
    reformat_error("ipyopt.Problem");
    return nullptr;
  }

  receive_members(self->member_slots, py_eval_f.owned, py_eval_grad_f.owned,
                  py_eval_g.owned, py_eval_jac_g.owned, py_eval_h.owned,
                  py_intermediate_callback.owned);

  Ipopt::TNLP *nlp;
  std::tie(nlp, self->nlp) =
      build_nlp(py_eval_f.callable, py_eval_grad_f.callable, py_eval_g.callable,
                py_eval_jac_g.callable, std::move(sparsity_indices_jac_g),
                py_eval_h.callable, std::move(sparsity_indices_h),
                py_intermediate_callback.callable, std::move(x_l),
                std::move(x_u), std::move(g_l), std::move(g_u));
  self->bundle->take_nlp(nlp);
  self->nlp->_x_scaling = std::move(x_scaling);
  self->nlp->_g_scaling = std::move(g_scaling);
  self->nlp->_obj_scaling = obj_scaling;
  if (is_null(py_eval_h.callable))
    self->bundle->without_hess();
  return (PyObject *)self;
}

static char IPYOPT_SOLVE_DOC[] = R"mdoc(
solve(x: numpy.ndarray[numpy.float64], mult_g: Optional[numpy.ndarray[numpy.float64]] = None, mult_x_L: Optional[numpy.ndarray[numpy.float64]] = None, mult_x_U: Optional[numpy.ndarray[numpy.float64]] = None) -> tuple[numpy.ndarray[numpy.float64], float, int]

Call Ipopt to solve problem created before and return
a tuple containing the final solution ``x``, the value of the final objective function
and the return status code of Ipopt.
For performance reasons, to avoid copying, the argument ``x`` is mutated by this function and it will be the same object as
the first element of the returned tuple.
To keep the initial value of ``x``, the user is responsible to make a copy before a call to this method.
``mult_g``, ``mult_x_L``, ``mult_x_U`` are optional keyword only arguments
allowing previous values of bound multipliers to be passed in warm
start applications.
If passed, these variables are modified.
)mdoc";

static PyObject *py_solve(PyObject *self, PyObject *args, PyObject *keywords) {
  auto *py_problem = (PyNlpApp *)self;
  PyArrayObject *py_mult_x_L = nullptr, *py_mult_x_U = nullptr,
                *py_mult_g = nullptr;
  PyArrayObject *py_x0 = nullptr;
  const char *arg_names[] = {"x0", "mult_g", "mult_x_L", "mult_x_U", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, keywords, "O!|$O!O!O!", const_cast<char **>(arg_names),
          &PyArray_Type, &py_x0, &PyArray_Type, &py_mult_g, &PyArray_Type,
          &py_mult_x_L, &PyArray_Type, &py_mult_x_U) ||
      !check_array_dim(py_x0, py_problem->nlp->n, "x0") ||
      (py_mult_g != nullptr &&
       !check_array_dim(py_mult_g, py_problem->nlp->m, "mult_g")) ||
      (py_mult_x_L != nullptr &&
       !check_array_dim(py_mult_x_L, py_problem->nlp->n, "mult_x_L")) ||
      (py_mult_x_U != nullptr &&
       !check_array_dim(py_mult_x_U, py_problem->nlp->n, "mult_x_U")))
    return nullptr;
  py_problem->nlp->set_initial_values(
      (double *)PyArray_DATA(py_x0), optional_array_data(py_mult_g),
      optional_array_data(py_mult_x_L), optional_array_data(py_mult_x_U));
  auto status = py_problem->bundle->optimize();
  if (PyErr_Occurred())
    return nullptr;
  Py_XINCREF(py_x0); // This is an existing object.
  // If we would not increase the ref counter here, a reference would get lost if the
  // tuple gets garbage collected.
  return py_tuple((PyObject *)py_x0,
                  PyFloat_FromDouble(py_problem->nlp->out_obj_value),
                  PyLong_FromLong(status));
}

static char IPYOPT_SET_OPTION_DOC[] = R"mdoc(
set(**kwargs) -> None

Set one or more Ipopt options. The Python type of the value objects have to match
the corresponding types (i.e. ``str``, ``float`` or ``int``) of the Ipopt options.
Refer to the Ipopt document for more information about Ipopt options, or use :func:`get_ipopt_options`
to see a list of available options.
)mdoc";
static PyObject *py_set(PyObject *self, PyObject *args, PyObject *keywords) {

  if (!check_kwargs(keywords) || !check_no_args("set", args) ||
      !set_options(*((PyNlpApp *)self)->bundle, keywords))
    return nullptr;
  Py_RETURN_NONE;
}

static char IPYOPT_SET_PROBLEM_SCALING_DOC[] = R"mdoc(
set_problem_scaling(obj_scaling: float, x_scaling: Optional[numpy.ndarray] = None, g_scaling: Optional[numpy.ndarray] = None) -> None

Set scaling parameters for the NLP.
Attention: Only takes effect if ``nlp_scaling_method="user-scaling"`` is set via :func:`set` or the ``ipopt_options`` argument!
If ``x_scaling`` or ``g_scaling`` is not specified or explicitly are ``None``, then no scaling for ``x`` resp. ``g`` is done.
This corresponds to the `TNLP::get_scaling_parameters`_ method.

.. _`TNLP::get_scaling_parameters`: https://coin-or.github.io/Ipopt/classIpopt_1_1TNLP.html#a3e840dddefbe48a048d213bd02b39854
)mdoc";

static PyObject *py_set_problem_scaling(PyObject *self, PyObject *args,
                                        PyObject *keywords) {
  double obj_scaling;
  std::vector<double> x_scaling, g_scaling;
  const char *arg_names[] = {"obj_scaling", "x_scaling", "g_scaling", nullptr};
  NlpData &nlp = *((PyNlpApp *)self)->nlp;
  if (!PyArg_ParseTupleAndKeywords(
          args, keywords, "d|O&O&:%s", const_cast<char **>(arg_names),
          &obj_scaling, &parse_vec<arg_x_scaling, true, double>, &x_scaling,
          &parse_vec<arg_g_scaling, true, double>, &g_scaling) ||
      !check_vec_size<double, true>(x_scaling, nlp.n,
                                    "%s() argument x_scaling") ||
      !check_vec_size<double, true>(g_scaling, nlp.m,
                                    "%s() argument g_scaling")) {
    reformat_error("ipyopt.Problem.set_problem_scaling");
    return nullptr;
  }
  nlp._x_scaling = std::move(x_scaling);
  nlp._g_scaling = std::move(g_scaling);
  nlp._obj_scaling = obj_scaling;
  Py_RETURN_NONE;
}

static PyObject *py_ipopt_type(IpoptOption::Type t) {
  PyObject *obj;
  switch (t) {
  case IpoptOption::Integer:
    obj = (PyObject *)&PyLong_Type;
  case IpoptOption::Number:
    obj = (PyObject *)&PyFloat_Type;
  case IpoptOption::String:
    obj = (PyObject *)&PyUnicode_Type;
  default:
    obj = Py_None;
  }
  Py_INCREF(obj);
  return obj;
}

static char GET_IPOPT_OPTIONS_DOC[] = R"mdoc(
get_ipopt_options() -> list[dict[str, Any]]

Get a list of all Ipopt options.
The items of the returned list are dicts, containing the fields::

    {
      "name": str,
      "type": Union[Type[int], Type[float], Type[str], None],
      "description_short": str,
      "description_long": str,
      "category": str
    }

)mdoc";

static PyObject *py_get_ipopt_options(PyObject *, PyObject *) {
  const auto options = get_ipopt_options();
  auto lst = PyList_New(options.size());
  auto i = std::size_t{0};
  for (const auto &opt : options) {
    auto *dict = py_dict(
        std::make_tuple("name", opt.name.data()),
        std::make_tuple("type", py_ipopt_type(opt.type)),
        std::make_tuple("description_short", opt.description_short.data()),
        std::make_tuple("description_long", opt.description_long.data()),
        std::make_tuple("category", opt.category.data()));
    PyList_SET_ITEM(lst, i++, dict);
  }
  return lst;
}

PyObject *py_get_stats(PyObject *self, void *) {
  auto nlp = ((PyNlpApp *)self)->nlp;
  return py_dict(
      std::make_tuple("n_eval_f", nlp->out_stats.n_eval_f),
      std::make_tuple("n_eval_grad_f", nlp->out_stats.n_eval_grad_f),
      std::make_tuple("n_eval_g_eq", nlp->out_stats.n_eval_g_eq),
      std::make_tuple("n_eval_jac_g_eq", nlp->out_stats.n_eval_jac_g_eq),
      std::make_tuple("n_eval_g_ineq", nlp->out_stats.n_eval_g_ineq),
      std::make_tuple("n_eval_jac_g_ineq", nlp->out_stats.n_eval_jac_g_ineq),
      std::make_tuple("n_eval_h", nlp->out_stats.n_eval_h),
      std::make_tuple("n_iter", nlp->out_stats.n_iter));
}

// Begin Python Module code section

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, .m_name = "ipyopt",
    .m_doc = "Python interface to Ipopt", .m_size = -1,
    .m_methods = (PyMethodDef[]){{"get_ipopt_options", py_get_ipopt_options,
                                  METH_NOARGS, GET_IPOPT_OPTIONS_DOC},
                                 {nullptr, nullptr, 0, nullptr}}};

PyMethodDef problem_methods[] = {
    {"solve", (PyCFunction)py_solve, METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR(IPYOPT_SOLVE_DOC)},
    {"set", (PyCFunction)py_set, METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR(IPYOPT_SET_OPTION_DOC)},
    {"set_problem_scaling", (PyCFunction)py_set_problem_scaling,
     METH_VARARGS | METH_KEYWORDS, PyDoc_STR(IPYOPT_SET_PROBLEM_SCALING_DOC)},
    {nullptr, nullptr, 0, nullptr},
};

static PyTypeObject IPyOptProblemType = {
    PyVarObject_HEAD_INIT(nullptr, 0) // ---
        .tp_name = "ipyopt.Problem",
    .tp_basicsize = sizeof(PyNlpApp),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)py_ipopt_problem_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_doc = PyDoc_STR(IPYOPT_PROBLEM_DOC),
    .tp_traverse = (traverseproc)py_ipopt_problem_traverse,
    .tp_clear = (inquiry)py_ipopt_problem_clear,
    .tp_methods = problem_methods,
    .tp_getset =
        (PyGetSetDef[]){{"stats", py_get_stats, nullptr,
                         "dict[str, int]: Stats about an optimization run",
                         nullptr},
                        {nullptr, nullptr, nullptr, nullptr, nullptr}},
    .tp_new = py_ipopt_problem_new};

PyMODINIT_FUNC PyInit_ipyopt(void) {
  // Finish initialization of the problem type
  if (PyType_Ready(&IPyOptProblemType) < 0)
    return nullptr;

  PyObject *module = PyModule_Create(&moduledef);

  if (module == nullptr)
    return nullptr;

  Py_INCREF(&IPyOptProblemType);
  if (PyModule_AddObject(module, "Problem", (PyObject *)&IPyOptProblemType) <
      0) {
    Py_DECREF(&IPyOptProblemType);
    Py_DECREF(module);
    return nullptr;
  }
#ifdef VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
  PyModule_AddUnicodeConstant(module, "__version__",
                              MACRO_STRINGIFY(VERSION_INFO));
#else
  PyModule_AddStringConstant(module, "__version__", "dev");
#endif

  // Initialize numpy (a segfault will occur if using numpy array without this)
  import_array();
  if (PyErr_Occurred())
    Py_FatalError("Unable to initialize module ipyopt");

  return module;
}
}
