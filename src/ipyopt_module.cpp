#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ipyopt_ARRAY_API
#include "numpy/arrayobject.h"

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
  PyObject *rows, *cols;
  Py_ssize_t n, i;
  if (!PyTuple_Check(obj)) {
    PyErr_Format(PyExc_TypeError,
                 "Sparsity info: a tuple of size 2 is needed.");
    return false;
  }
  if (PyTuple_Size(obj) != 2) {
    PyErr_Format(
        PyExc_TypeError,
        "Sparsity info: a tuple of size 2 is needed. Found tuple of size %d",
        PyTuple_Size(obj));
    return false;
  }
  rows = PyTuple_GetItem(obj, 0);
  cols = PyTuple_GetItem(obj, 1);
  n = PyObject_Length(rows);
  if (n != PyObject_Length(cols)) {
    PyErr_Format(PyExc_TypeError,
                 "Sparsity info: length of row indices (%d) does not match "
                 "lenth of column indices (%d)",
                 n, PyObject_Length(cols));
    return false;
  }
  std::vector<int> row, col;
  PyObject *row_iter = PyObject_GetIter(rows);
  PyObject *col_iter = PyObject_GetIter(cols);
  PyObject *row_item, *col_item;
  for (i = 0; i < n; i++) {
    row_item = PyIter_Next(row_iter);
    col_item = PyIter_Next(col_iter);
    if (row_item != nullptr)
      row.push_back(PyLong_AsLong(row_item));
    if (col_item != nullptr)
      col.push_back(PyLong_AsLong(col_item));
    if (row_item == nullptr || col_item == nullptr ||
        PyErr_Occurred() != nullptr) {
      PyErr_Format(PyExc_TypeError,
                   "Sparsity info: Row an column indices must be integers");
      return false;
    }
  }
  idx = std::make_tuple(row, col);
  return true;
}
static bool check_callback(PyObject *obj, const char *name) {
  if (PyCallable_Check(obj))
    return true;
  PyErr_Format(PyExc_TypeError,
               "Need a callable object for callback function %s", name);
  return false;
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
      PyErr_Format(PyExc_ValueError, "Failed to set the IPOpt option '%s'",
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
constexpr std::size_t max_owned_py_objects = 6;

extern "C" {
typedef struct {
  PyObject_HEAD NlpBundle *bundle;
  NlpData *nlp;
  // Python memory management:
  PyObject *owned_py_objects[max_owned_py_objects];
} PyNlpApp;

static int py_ipopt_problem_clear(PyNlpApp *self) {
  for (std::size_t i = 0; i < max_owned_py_objects; i++)
    if (self->owned_py_objects[i] != nullptr) {
      Py_CLEAR(self->owned_py_objects[i]);
      self->owned_py_objects[i] = nullptr;
    }
  return 0;
}
static void py_ipopt_problem_dealloc(PyObject *self) {
  auto obj = (PyNlpApp *)self;

  PyObject_GC_UnTrack(self);
  py_ipopt_problem_clear(obj);
  if (obj->bundle != nullptr) {
    delete obj->bundle;
    obj->bundle = nullptr;
  }
  Py_TYPE(self)->tp_free(self);
}
static int py_ipopt_problem_traverse(PyNlpApp *self, visitproc visit,
                                     void *arg) {
  for (std::size_t i = 0; i < max_owned_py_objects; i++)
    if (self->owned_py_objects[i] != nullptr) {
      Py_VISIT(self->owned_py_objects[i]);
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
    IPOpt problem type in python

    Problem(n: int, x_l: numpy.ndarray[numpy.float64], x_u: numpy.ndarray[numpy.float64], m: int, g_l: numpy.ndarray[numpy.float64], g_u: numpy.ndarray[numpy.float64], sparsity_indices_jac_g: Tuple[Sequence[float], Sequence[float]], sparsity_indices_h: Tuple[Sequence[float], Sequence[float]], eval_f: Callable, eval_grad_f: Callable, eval_g: Callable, eval_jac_g: Callable, eval_h: Optional[Callable] = None, intermediate_callback: Optional[Callable] = None, obj_scaling: float = 1., x_scaling: Optional[numpy.ndarray[numpy.float64]] = None, g_scaling: Optional[numpy.ndarray[numpy.float64]] = None, ipopt_options: Optional[Dict[str, Union[int, float, str]]] = None) -> Problem

    n  -- Number of variables (dimension of x)
    x_l -- Lower bound of x as bounded constraints
    x_u -- Upper bound of x as bounded constraints
        both xL, xU should be one 1-dim arrays with length n

    m  -- Number of constraints
    g_l -- Lower bound of constraints
    g_u -- Upper bound of constraints
        both gL, gU should be one dimension arrays with length m
    sparsity_indices_jac_g -- Positions of non-zero entries of jac_g in the form of a tuple of two sequences of the same length (first list are column indices, second column are row indices)
    sparsity_indices_h -- Positions of non-zero entries of hess
    eval_f -- Callback function to calculate objective value.
        Signature: `eval_f(x: numpy.ndarray) -> float`,
    eval_grad_f -- calculates gradient for objective function.
        Signature: `eval_grad_f(x: numpy.ndarray, out: numpy.ndarray) -> Any`. 
        The array `out` must be a 1-dim array matching the length of `x`, i.e. `n`.
        A possible return value will be ignored.
    eval_g -- calculates the constraint values and return an array
        Signature: `eval_g(x: numpy.ndarray, out: numpy.ndarray) -> Any`.
        The array `out` must be a 1-dim array of length `m`.
        A possible return value will be ignored.
    eval_jac_g -- calculates the Jacobi matrix.
        Signature: `eval_jac_g(x: numpy.ndarray, out: numpy.ndarray) -> Any`. The array `out` must be a 1-dim array whose entries are the entries of the Jacobi matrix jac_g listed in `sparsity_indices_jac_g` (order matters).
        A possible return value will be ignored.
    eval_h -- calculates the hessian matrix (optional).
        Signature: `eval_h(x: numpy.ndarray, lagrange: numpy.ndarray, obj_factor: numpy.ndarray, out: numpy.ndarray) -> Any`.
        The array `out` must be a 1-dim array and contain the entries of
        `obj_factor * Hess(f) + lagrange[i] * Hess(g[i])` (sum over `i`),
        listed in `sparsity_indices_hess` for given `obj_factor: float`
        and `lagrange: numpy.ndarray` of shape (m,).
        A possible return value will be ignored.
        If omitted, the parameter sparsity_indices_hess will be ignored and Ipopt will use approximated hessian
        which will make the convergence slower.
    intermediate_callback --  Intermediate Callback method for the user.
        This method is called once per iteration (during the convergence check), and can be used to obtain information about the optimization status while Ipopt solves the problem, and also to request a premature termination (see the IpOpt docs for more details).
        Signature: `intermediate_callback(mode: int, iter: int, obj_value: float, inf_pr: float, inf_du: float, mu: float, d_norm: float, regularization_size: float, alpha_du: float, alpha_pr: float) -> Any`.
    obj_scaling -- A scaling factor for the objective value (see `set_problem_scaling`).
    x_scaling   -- Either None (no scaling) or a numpy.ndarray of length n, scaling the x variables (see `set_problem_scaling`).
    g_scaling   -- Either None (no scaling) or a numpy.ndarray of length m, scaling the g variables (see `set_problem_scaling`).
    ipopt_options -- A dict of key value pairs, to be passed to IPOpt (see ipopt --print-options or the IPOpt manual)"
)mdoc";
static PyObject *py_ipopt_problem_new(PyTypeObject *type, PyObject *args,
                                      PyObject *keywords) {
  auto *self = (PyNlpApp *)type->tp_alloc(type, 0);
  for (std::size_t i = 0; i < max_owned_py_objects; i++)
    self->owned_py_objects[i] = nullptr;
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
  PyObject *py_eval_f = nullptr;
  PyObject *py_eval_grad_f = nullptr;
  PyObject *py_eval_g = nullptr;
  PyObject *py_eval_jac_g = nullptr;
  PyObject *py_eval_h = nullptr;
  PyObject *py_intermediate_callback = nullptr;
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
          "iO&O&iO&O&OOOOOO|OOdO&O&O:%s", // function name will be substituted later
          const_cast<char **>(arg_names), &n,
          &parse_vec<arg_x_l, false, double>, &x_l,
          &parse_vec<arg_x_u, false, double>, &x_u, &m,
          &parse_vec<arg_g_l, false, double>, &g_l,
          &parse_vec<arg_g_u, false, double>, &g_u, &py_sparsity_indices_jac_g,
          &py_sparsity_indices_h, &py_eval_f, &py_eval_grad_f, &py_eval_g,
          &py_eval_jac_g, &py_eval_h, &py_intermediate_callback, &obj_scaling,
          &parse_vec<arg_x_scaling, true, double>, &x_scaling,
          &parse_vec<arg_g_scaling, true, double>, &g_scaling,
          &py_ipopt_options) ||
      !parse_sparsity_indices(py_sparsity_indices_jac_g,
                              sparsity_indices_jac_g) ||
      !check_callback(py_eval_f, "eval_f") ||
      !check_callback(py_eval_grad_f, "eval_grad_f") ||
      !check_callback(py_eval_g, "eval_g") ||
      !check_callback(py_eval_jac_g, "eval_jac_g") ||
      !check_non_negative(n, "n") || !check_non_negative(m, "m") ||
      !check_vec_size<double, false>(x_l, n, "%s() argument x_L") ||
      !check_vec_size<double, false>(x_u, n, "%s() argument x_U") ||
      !check_vec_size<double, false>(g_l, m, "%s() argument g_L") ||
      !check_vec_size<double, false>(g_u, m, "%s() argument g_U") ||
      !(py_eval_h == nullptr ||
        (check_callback(py_eval_h, "h") &&
         parse_sparsity_indices(py_sparsity_indices_h, sparsity_indices_h))) ||
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

  PyObject *owned_py_objects[max_owned_py_objects] = {
      py_eval_f,     py_eval_grad_f, py_eval_g,
      py_eval_jac_g, py_eval_h,      py_intermediate_callback};
  for (std::size_t i = 0; i < max_owned_py_objects; i++) {
    self->owned_py_objects[i] = owned_py_objects[i];
    if (owned_py_objects[i] != nullptr)
      Py_XINCREF(owned_py_objects[i]);
  }
  auto nlp =
      new NlpBase{ipyopt::py::F{py_eval_f},
                  ipyopt::py::GradF{py_eval_grad_f},
                  ipyopt::py::G{py_eval_g},
                  ipyopt::py::JacG{py_eval_jac_g},
                  std::move(sparsity_indices_jac_g),
                  ipyopt::py::H{py_eval_h},
                  std::move(sparsity_indices_h),
                  ipyopt::py::IntermediateCallback{py_intermediate_callback},
                  std::move(x_l),
                  std::move(x_u),
                  std::move(g_l),
                  std::move(g_u)};
  self->nlp = nlp;
  self->bundle->take_nlp(nlp);
  self->nlp->_x_scaling = std::move(x_scaling);
  self->nlp->_g_scaling = std::move(g_scaling);
  self->nlp->_obj_scaling = obj_scaling;
  if (py_eval_h == nullptr)
    self->bundle->without_hess();
  return (PyObject *)self;
}

static char IPYOPT_SOLVE_DOC[] = R"mdoc(
solve(x: numpy.ndarray[numpy.float64], *, mult_g: Optional[numpy.ndarray[numpy.float64]] = None, mult_x_L: Optional[numpy.ndarray[numpy.float64]] = None, mult_x_U: Optional[numpy.ndarray[numpy.float64]] = None) -> Tuple[numpy.ndarray[numpy.float64], float, int]

Call Ipopt to solve problem created before and return
a tuple containing the final solution x, the value of the final objective function
and the return status code of ipopt.
mult_g, mult_x_L, mult_x_U are optional keyword only arguments
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
  return py_tuple((PyObject *)py_x0,
                  PyFloat_FromDouble(py_problem->nlp->out_obj_value),
                  PyLong_FromLong(status));
}

static char IPYOPT_SET_OPTION_DOC[] = R"mdoc(
set(**kwargs)

Set one or more Ipopt options. The python type of the value objects have to match
the corresponding types (i.e. str, float or int) of the IPOpt options.
Refer to the Ipopt document for more information about Ipopt options, or use
ipopt --print-options
to see a list of available options.
)mdoc";
static PyObject *py_set(PyObject *self, PyObject *args, PyObject *keywords) {

  if (!check_kwargs(keywords) || !check_no_args("set", args) ||
      !set_options(*((PyNlpApp *)self)->bundle, keywords))
    return nullptr;
  Py_RETURN_NONE;
}

static char IPYOPT_SET_PROBLEM_SCALING_DOC[] = R"mdoc(
set_problem_scaling(obj_scaling: float, x_scaling: Optional[numpy.ndarray] = None, g_scaling: Optional[numpy.ndarray] = None)

Set scaling parameters for the NLP.
Attention: Only takes effect if `nlp_scaling_method="user-scaling"` is set via `Problem.set` or `ipopt_options`!
If x_scaling or g_scaling is not specified or explicitly are None, then no scaling for x resp. g is done.
This corresponds to the TNLP::get_scaling_parameters method.
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

// Begin Python Module code section

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "ipyopt",
    .m_doc = "A hook between Ipopt and Python",
    .m_size = -1,
    .m_methods = (PyMethodDef[]){{nullptr, nullptr, 0, nullptr}},
    .m_slots = nullptr,
    .m_traverse = nullptr,
    .m_clear = nullptr,
    .m_free = nullptr,
};

PyMethodDef problem_methods[] = {
    {"solve", (PyCFunction)py_solve, METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR(IPYOPT_SOLVE_DOC)},
    {"set", (PyCFunction)py_set, METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR(IPYOPT_SET_OPTION_DOC)},
    {"set_problem_scaling", (PyCFunction)py_set_problem_scaling,
     METH_VARARGS | METH_KEYWORDS, PyDoc_STR(IPYOPT_SET_PROBLEM_SCALING_DOC)},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject IPyOptProblemType = {
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "ipyopt.Problem",
    .tp_basicsize = sizeof(PyNlpApp),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)py_ipopt_problem_dealloc,
    .tp_getattr = 0,
    .tp_setattr = 0,
    .tp_as_async = 0,
    .tp_repr = 0,
    .tp_as_number = 0,
    .tp_as_sequence = 0,
    .tp_as_mapping = 0,
    .tp_hash = 0,
    .tp_call = 0,
    .tp_str = 0,
    .tp_getattro = 0,
    .tp_setattro = 0,
    .tp_as_buffer = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_doc = PyDoc_STR(IPYOPT_PROBLEM_DOC),
    .tp_traverse = (traverseproc)py_ipopt_problem_traverse,
    .tp_clear = (inquiry)py_ipopt_problem_clear,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = problem_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = 0,
    .tp_alloc = 0,
    .tp_new = py_ipopt_problem_new};

PyMODINIT_FUNC PyInit_ipyopt(void) {
  PyObject *module;
  // Finish initialization of the problem type
  if (PyType_Ready(&IPyOptProblemType) < 0)
    return nullptr;

  module = PyModule_Create(&moduledef);

  if (module == nullptr)
    return nullptr;

  Py_INCREF(&IPyOptProblemType);
  PyModule_AddObject(module, "Problem", (PyObject *)&IPyOptProblemType);
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
