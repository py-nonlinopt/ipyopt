/*
 * Copyright (c) 2008, Eric You Xu, Washington University All rights
 * reserved. Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following conditions
 * are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the Washington University nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ipyopt_ARRAY_API
#include "numpy/arrayobject.h"

#include "callback.h"

#ifndef SAFE_FREE
#define SAFE_FREE(p) {if(p) {free(p); (p)= NULL;}}
#endif


static _Bool parse_sparsity_indices(PyObject* obj, SparsityIndices *idx);
static void sparsity_indices_free(SparsityIndices* idx);
static Bool check_type(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name);
static Bool check_optional(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name);
static Bool check_callback(PyObject *callback, const char *name);
static Bool _PyArray_Check(const PyObject* obj) { return PyArray_Check(obj); } // Macro -> function
static Bool _PyLong_Check(const PyObject* obj) { return PyLong_Check(obj); } // Macro -> function
static Bool _PyUnicode_Check(const PyObject* obj) { return PyUnicode_Check(obj); } // Macro -> function
static Bool _PyFloat_Check(const PyObject* obj) { return PyFloat_Check(obj); } // Macro -> function
static Bool check_kwargs(const PyObject *kwargs);
static Bool check_non_negative(int n, const char *name) {
  if(n >= 0) return TRUE;
  PyErr_Format(PyExc_ValueError, "%s can't be negative", name);
  return FALSE;
}
static Bool array_copy_data(PyArrayObject *arr, Number **dest) {
  unsigned int n = PyArray_SIZE(arr), i;
  Number *data = PyArray_DATA(arr);
  *dest = malloc(n * sizeof(Number*));
  if(!*dest) {
    PyErr_NoMemory();
    return FALSE;
  }
  for(i=0; i<n; i++)
    (*dest)[i] = data[i];
  return TRUE;
}
static Bool check_array_dim(PyArrayObject *arr, unsigned int dim, const char *name) {
  if((unsigned int)PyArray_NDIM(arr) != 1) {
    PyErr_Format(PyExc_ValueError, "%s has wrong number of dimensions. Expected %d, got %d", name, 1, PyArray_NDIM(arr));
    return FALSE;
  }
  if(PyArray_DIMS(arr)[0] == dim) return TRUE;
  PyErr_Format(PyExc_ValueError, "%s has wrong shape. Expected (%d,), found (%d,)", name, dim, PyArray_DIMS(arr)[0]);
  return FALSE;
}

typedef struct {
  PyObject_HEAD IpoptProblem nlp;
  DispatchData data;
  Index py_n;
  Index py_m;
} IPyOptProblemObject;


static Bool set_int_option(IpoptProblem nlp, char *key, PyObject *obj)
{ return AddIpoptIntOption(nlp, key, PyLong_AsLong(obj)); }
static Bool set_num_option(IpoptProblem nlp, char *key, PyObject *obj)
{ return AddIpoptNumOption(nlp, key, PyFloat_AsDouble(obj)); }
static Bool set_str_option(IpoptProblem nlp, char *key, PyObject *obj)
{ return AddIpoptStrOption(nlp, key, (char*)PyUnicode_AsUTF8(obj)); }

typedef struct {
  Bool (*check)(const PyObject*);
  Bool (*set_option)(IpoptProblem, char*, PyObject*);
  char *type_repr;
} type_mapping_record;

const static type_mapping_record type_mapping[] = {
  {.check = _PyFloat_Check, .set_option = set_num_option, .type_repr = "num"},
  {.check = _PyUnicode_Check, .set_option = set_str_option, .type_repr = "str"},
  {.check = _PyLong_Check, .set_option = set_int_option, .type_repr = "int"}
};

static Bool set_option(IpoptProblem nlp, PyObject *key, PyObject *val) {
  const char *c_key = PyUnicode_AsUTF8(key);
  unsigned int i;
  Bool ret = FALSE;
  for(i=0; i<sizeof(type_mapping)/sizeof(type_mapping_record); i++)
    if((*type_mapping[i].check)(val)) {
      ret = (*type_mapping[i].set_option)(nlp, (char*)c_key, val);
      if(PyErr_Occurred() != NULL) return FALSE;
      if(!ret) {
        PyErr_Format(PyExc_ValueError, "%s is not a valid %s option", c_key, type_mapping[i].type_repr);
        return FALSE;
      }
      return TRUE;
    }
  PyErr_Format(PyExc_TypeError, "The value for option %s has unsupported type", c_key);
  return FALSE;
}
static Bool set_options(IpoptProblem nlp, PyObject *dict) {
  PyObject *key, *val;
  Py_ssize_t pos = 0;

  if(dict == NULL) return TRUE;
  while (PyDict_Next(dict, &pos, &key, &val)) {
    if(!set_option(nlp, key, val))
      return FALSE;
  }
  return TRUE;
}

static Bool check_callback(PyObject *obj, const char *name) {
  if(PyCallable_Check(obj)) return TRUE;
  PyErr_Format(PyExc_TypeError, "Need a callable object for callback function %s", name);
  return FALSE;
}

static Bool check_no_args(const char* f_name, PyObject *args) {
  if(args == NULL) return TRUE;
  if(!PyTuple_Check(args)) {
    PyErr_Format(PyExc_RuntimeError, "Argument keywords is not a tuple");
    return FALSE;
  }
  unsigned int n = PyTuple_Size(args);
  if(n == 0) return TRUE;
  PyErr_Format(PyExc_TypeError, "%s() takes 0 positional arguments but %d %s given", f_name, n, n==1?"was":"were");
  return FALSE;
}

static char IPYOPT_SET_OPTION_DOC[] =
  "set(**kwargs)" "\n\n"
  "Set one or more Ipopt options. The python type of the value objects have to match "
  "the corresponding types (i.e. str, float or int) of the IPOpt options. "
  "Refer to the Ipopt" "\n"
  "document for more information about Ipopt options, or use" "\n"
  "ipopt --print-options" "\n"
  "to see a list of available options.";
static PyObject *set(PyObject *self, PyObject *args, PyObject *keywords) {
  IpoptProblem nlp = (IpoptProblem)(((IPyOptProblemObject*)self)->nlp);

  if(!check_kwargs(keywords) || !check_no_args("set", args)) return PyErr_Occurred();
  if(!set_options(nlp, keywords))
    return NULL;
  Py_RETURN_NONE;
}

static char IPYOPT_SET_PROBLEM_SCALING_DOC[] =
  "set_problem_scaling(obj_scaling: float, x_scaling: Optional[numpy.ndarray] = None, g_scaling: Optional[numpy.ndarray] = None)"
  "\n\n"
  "Set scaling parameters for the NLP." "\n"
  "Attention: Only takes effect if `nlp_scaling_method=\"user-scaling\"` is set via `Problem.set` or `ipopt_options`!" " "
  "If x_scaling or g_scaling is not specified or explicitly are None, then no scaling for x resp. g is done. "
  "This corresponds to the TNLP::get_scaling_parameters method. "
  ;
static _Bool _set_problem_scaling(PyObject *self, double obj_scaling, PyArrayObject *py_x_scaling, PyArrayObject *py_g_scaling) {
  IPyOptProblemObject* py_problem = (IPyOptProblemObject*)self;
  IpoptProblem nlp = (IpoptProblem)(py_problem->nlp);
  if(!(py_x_scaling == NULL || (PyObject*)py_x_scaling == Py_None || check_array_dim(py_x_scaling, py_problem->py_n, "x_scaling"))
     || !(py_g_scaling == NULL || (PyObject*)py_g_scaling == Py_None || check_array_dim(py_g_scaling, py_problem->py_m, "g_scaling")))
    return NULL;
  
  Bool result = SetIpoptProblemScaling(nlp, obj_scaling,
                                       (py_x_scaling == NULL || (PyObject*)py_x_scaling == Py_None)?NULL:PyArray_DATA(py_x_scaling),
                                       (py_g_scaling == NULL || (PyObject*)py_g_scaling == Py_None)?NULL:PyArray_DATA(py_g_scaling));
  return result;
}

static PyObject *set_problem_scaling(PyObject *self, PyObject *args, PyObject *keywords) {
  double obj_scaling;
  PyObject *py_x_scaling = NULL;
  PyObject *py_g_scaling = NULL;
  if(!PyArg_ParseTupleAndKeywords(args, keywords, "d|OO:ipyopt.Problem.set_problem_scaling",
				  (char*[]){"obj_scaling", "x_scaling", "g_scaling", NULL},
                                  &obj_scaling,
                                  &py_x_scaling,
                                  &py_g_scaling)
     || !check_optional(py_x_scaling, _PyArray_Check, "x_scaling", "Optional[numpy.ndarray]")
     || !check_optional(py_g_scaling, _PyArray_Check, "g_scaling", "Optional[numpy.ndarray]")
     || !_set_problem_scaling(self, obj_scaling, (PyArrayObject*)py_x_scaling, (PyArrayObject*)py_g_scaling))
    return NULL;
  Py_RETURN_FALSE;
}

static char IPYOPT_SOLVE_DOC[] = "solve(x: numpy.ndarray[numpy.float64], *, mult_g: Optional[numpy.ndarray[numpy.float64]] = None, mult_x_L: Optional[numpy.ndarray[numpy.float64]] = None, mult_x_U: Optional[numpy.ndarray[numpy.float64]] = None) -> Tuple[numpy.ndarray[numpy.float64], float, int]"
  "\n\n"
  "Call Ipopt to solve problem created before and return" "\n"
  "a tuple containing the final solution x, the value of the final objective function" "\n"
  "and the return status code of ipopt." "\n\n"
  "mult_g, mult_x_L, mult_x_U are optional keyword only arguments" "\n"
  "allowing previous values of bound multipliers to be passed in warm" "\n"
  "start applications."
  "If passed, these variables are modified.";

static PyObject *solve(PyObject *self, PyObject *args, PyObject *keywords) {
  enum ApplicationReturnStatus status;	// Solve return code
  int i;
  
  // Return values
  IPyOptProblemObject *py_problem = (IPyOptProblemObject*)self;
  int n = py_problem->py_n;
  
  IpoptProblem nlp = py_problem->nlp;
  if(nlp == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "nlp objective passed to solve is NULL\n Problem created?\n");
    return NULL;
  }
  DispatchData *callbacks = (DispatchData*)&py_problem->data;
  int m = py_problem->py_m;
  
  npy_intp dX[1];
  
  PyArrayObject *x = NULL, *mL = NULL, *mU = NULL, *lambda = NULL;
  double *mL_data = NULL, *mU_data = NULL, *lambda_data = NULL;
  Number obj; // objective value
  
  PyObject *retval = NULL;
  PyArrayObject *x0 = NULL;
  
  Number *x_working = NULL;
  
  if(!PyArg_ParseTupleAndKeywords(args, keywords, "O!|$O!O!O!",
				  (char*[]){"x0", "mult_g", "mult_x_L", "mult_x_U", NULL},
				  &PyArray_Type, &x0,
				  &PyArray_Type, &lambda, // mult_g 
				  &PyArray_Type, &mL, // mult_x_L
				  &PyArray_Type, &mU) // mult_x_Y
     || !check_type((PyObject*)x0, &_PyArray_Check, "x0", "numpy.ndarray")
     || !check_array_dim(x0, n, "x0")
     || (mL && !check_array_dim(mL, n, "mL"))
     || (mU && !check_array_dim(mU, n, "mU"))
     || (lambda && !check_array_dim(lambda, m, "lambda"))
     || !array_copy_data(x0, &x_working)
     ) {
    SAFE_FREE(x_working);
    return NULL;
  }
  if(callbacks->py_eval_h == NULL)
    AddIpoptStrOption(nlp, "hessian_approximation", "limited-memory");
  
  // allocate space for the initial point and set the values
  dX[0] = n;
  
  x = (PyArrayObject*)PyArray_SimpleNew(1, dX, NPY_DOUBLE);
  // Allocate multiplier arrays
  if(mL == NULL) mL_data = malloc(n * sizeof(double));
  else mL_data = PyArray_DATA(mL);
  if(mU == NULL) mU_data = malloc(n * sizeof(double));
  else mU_data = PyArray_DATA(mU);
  if(lambda == NULL) lambda_data = malloc(m * sizeof(double));
  else lambda_data = PyArray_DATA(lambda);
  
  // For status code, see IpReturnCodes_inc.h in Ipopt
  if(x != NULL && mL_data != NULL && mU_data != NULL && lambda_data != NULL) {
    status = IpoptSolve(nlp, x_working, NULL, &obj,
                        lambda_data, mL_data, mU_data,
                        (UserDataPtr)callbacks);
    double *return_x_data = PyArray_DATA(x);
    for(i=0; i<n; i++)
      return_x_data[i] = x_working[i];
    if(!PyErr_Occurred())
      retval = PyTuple_Pack(3, PyArray_Return(x), PyFloat_FromDouble(obj), PyLong_FromLong(status));
  } else {
    Py_XDECREF(x);
    PyErr_NoMemory();
  }
  // clean up and return
  if(lambda == NULL && lambda_data != NULL) free(lambda_data);
  if(mU == NULL && mU_data != NULL) free(mU_data);
  if(mL == NULL && mL_data != NULL) free(mL_data);

  SAFE_FREE(x_working);
  Py_XDECREF(x);
  return retval;
}

static _Bool _set_intermediate_callback(PyObject *self, PyObject *py_intermediate_callback) {
  IPyOptProblemObject *py_problem = (IPyOptProblemObject*)self;
  IpoptProblem nlp = py_problem->nlp;
  DispatchData *callback_data = (DispatchData*)&py_problem->data;
  
  if(py_intermediate_callback == Py_None)
    py_intermediate_callback = NULL;
  if(py_intermediate_callback != NULL && !check_callback(py_intermediate_callback, "intermediate_callback"))
    return FALSE;
  if(callback_data->py_intermediate_callback != NULL)
    Py_XDECREF(callback_data->py_intermediate_callback);
  callback_data->py_intermediate_callback = py_intermediate_callback;
  if(py_intermediate_callback != NULL) {
    SetIntermediateCallback(nlp, intermediate_callback);
    Py_XINCREF(py_intermediate_callback);
  } else SetIntermediateCallback(nlp, NULL);
      
  // Put a Python function object into this data structure
  return TRUE;
}

static char IPYOPT_SET_INTERMEDIATE_CALLBACK_DOC[] =
  "set_intermediate_callback(callback_function: Optional[Callable])" "\n\n"
  "Set the intermediate callback function. "
  "This gets called each iteration." "\n"
  "For more info regarding the signature of the callback, see the doc of ipopt::Problem.";
static PyObject *set_intermediate_callback(PyObject *self, PyObject *args) {
  PyObject *py_intermediate_callback = NULL;
  
  if(!PyArg_ParseTuple(args, "O", &py_intermediate_callback)
     || !_set_intermediate_callback(self, py_intermediate_callback))
    return NULL;
  Py_RETURN_NONE;
}

static _Bool ipopt_problem_c_init(IPyOptProblemObject *object,
				  int n, Number *x_L, Number *x_U,
				  int m, Number *g_L, Number *g_U,
				  const DispatchData *callback_data) {
  int C_indexstyle = 0;
  IpoptProblem thisnlp = CreateIpoptProblem(n,
					    x_L, x_U, m, g_L, g_U,
					    callback_data->sparsity_indices_jac_g.n,
					    callback_data->sparsity_indices_hess.n,
					    C_indexstyle,
					    &eval_f, &eval_g,
					    &eval_grad_f,
					    &eval_jac_g, &eval_h);
  if(thisnlp == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Cannot create IpoptProblem instance");
    return FALSE;
  }
  object->py_n = n;
  object->py_m = m;
  object->nlp = thisnlp;
  memcpy((void*)&object->data, (void*)callback_data, sizeof(DispatchData));
  return TRUE;
}

static char IPYOPT_PROBLEM_DOC[] =
  "IPOpt problem type in python" "\n\n"
  "Problem(n: int, xL: numpy.ndarray[numpy.float64], xU: numpy.ndarray[numpy.float64], m: int, gL: numpy.ndarray[numpy.float64], gU: numpy.ndarray[numpy.float64], sparsity_indices_jac_g: Tuple[Sequence[float], Sequence[float]], sparsity_indices_hess: Tuple[Sequence[float], Sequence[float]], eval_f: Callable, eval_grad_f: Callable, eval_g: Callable, eval_jac_g: Callable, eval_h: Optional[Callable] = None, intermediate_callback: Optional[Callable] = None, obj_scaling: float = 1., x_scaling: Optional[numpy.ndarray[numpy.float64]] = None, g_scaling: Optional[numpy.ndarray[numpy.float64]] = None, ipopt_options: Optional[Dict[str, Union[int, float, str]]] = None) -> Problem" "\n\n"
  "n  -- Number of variables (dimension of x)" "\n"
  "xL -- Lower bound of x as bounded constraints" "\n"
  "xU -- Upper bound of x as bounded constraints" "\n"
  "\t" "both xL, xU should be one 1-dim arrays with length n" "\n\n"
  "m  -- Number of constraints" "\n"
  "gL -- Lower bound of constraints" "\n"
  "gU -- Upper bound of constraints" "\n"
  "\t" "both gL, gU should be one dimension arrays with length m" "\n"
  "sparsity_indices_jac_g -- Positions of non-zero entries of jac_g in the form of a tuple of two sequences of the same length (first list are column indices, second column are row indices)" "\n"
  "sparsity_indices_hess -- Positions of non-zero entries of hess" "\n"
  "eval_f -- Callback function to calculate objective value." "\n"
  "\t" "Signature: `eval_f(x: numpy.array) -> float`," "\n"
  "eval_grad_f -- calculates gradient for objective function." "\n"
  "\t" "Signature: `eval_grad_f(x: numpy.array, out: numpy.array) -> Any`. " "\n"
  "\t" "The array `out` must be a 1-dim array matching the length of `x`, i.e. `n`." "\n"
  "\t" "A possible return value will be ignored." "\n"
  "eval_g -- calculates the constraint values and return an array" "\n"
  "\t" "Signature: `eval_g(x: numpy.array, out: numpy.array) -> Any`." "\n"
  "\t" "The array `out` must be a 1-dim array of length `m`." "\n"
  "\t" "A possible return value will be ignored." "\n"
  "eval_jac_g -- calculates the Jacobi matrix." "\n"
  "\t" "Signature: `eval_jac_g(x: numpy.array, out: numpy.array) -> Any`. The array `out` must be a 1-dim array whose entries are the entries of the Jacobi matrix jac_g listed in `sparsity_indices_jac_g` (order matters)." "\n"
  "\t" "A possible return value will be ignored." "\n"
  "eval_h -- calculates the hessian matrix (optional)." "\n"
  "\t" "Signature: `eval_h(x: numpy.array, lagrange: numpy.array, obj_factor: numpy.array, out: numpy.array) -> Any`." "\n"
  "\t" "The array `out` must be a 1-dim array and contain the entries of" "\n"
  "\t" "`obj_factor * Hess(f) + lagrange[i] * Hess(g[i])` (sum over `i`)," "\n"
  "\t" "listed in `sparsity_indices_hess` for given `obj_factor: float`" "\n"
  "\t" "and `lagrange: numpy.array` of shape (m,)." "\n"
  "\t" "A possible return value will be ignored." "\n"
  "\t" "If omitted, the parameter sparsity_indices_hess will be ignored and Ipopt will use approximated hessian" "\n"
  "\t" "which will make the convergence slower." "\n"
  "intermediate_callback --  Intermediate Callback method for the user. This method is called once per iteration (during the convergence check), and can be used to obtain information about the optimization status while Ipopt solves the problem, and also to request a premature termination (see the IpOpt docs for more details). Signature: `intermediate_callback(mode: int, iter: int, obj_value: float, inf_pr: float, inf_du: float, mu: float, d_norm: float, regularization_size: float, alpha_du: float, alpha_pr: float) -> Any`." "\n"
  "obj_scaling -- A scaling factor for the objective value (see `set_problem_scaling`)." "\n"
  "x_scaling   -- Either None (no scaling) or a numpy.array of length n, scaling the x variables (see `set_problem_scaling`)." "\n"
  "g_scaling   -- Either None (no scaling) or a numpy.array of length m, scaling the g variables (see `set_problem_scaling`)." "\n"
  "ipopt_options -- A dict of key value pairs, to be passed to IPOpt (see ipopt --print-options or the IPOpt manual)";

static PyObject *py_ipopt_problem_new(PyTypeObject *type, PyObject *args, PyObject *keywords) {
  IPyOptProblemObject *self = NULL;

  DispatchData callback_data = {
    .py_eval_f = NULL,
    .py_eval_grad_f = NULL,
    .py_eval_g = NULL,
    .py_eval_jac_g = NULL,
    .py_eval_h = NULL,
    .py_intermediate_callback = NULL,
    .sparsity_indices_jac_g = { 0 },
    .sparsity_indices_hess = { 0 }
  };
  int n;			///< Number of variables
  PyArrayObject *py_x_L = NULL;
  PyArrayObject *py_x_U = NULL;
  int m;			///< Number of constraints
  PyArrayObject *py_g_L = NULL;
  PyArrayObject *py_g_U = NULL;
  
  Number *x_L = NULL;	///< lower bounds on x
  Number *x_U = NULL;	///< upper bounds on x
  Number *g_L = NULL;	///< lower bounds on g
  Number *g_U = NULL;	///< upper bounds on g
  Number obj_scaling = 1.;
  PyObject *py_x_scaling = NULL;
  PyObject *py_g_scaling = NULL;
  
  PyObject *py_sparsity_indices_jac_g = NULL;
  PyObject *py_sparsity_indices_hess = NULL;
  PyObject *py_ipopt_options = NULL;
  PyObject *py_intermediate_callback = NULL;
  
  if(!PyArg_ParseTupleAndKeywords(args, keywords, "iO!O!iO!O!OOOOOO|OOdOOO:ipyopt.Problem",
				  (char*[]){"n", "xL", "xU", "m", "gL", "gU", "sparsity_indices_jac_g", "sparsity_indices_hess", "eval_f", "eval_grad_f", "eval_g", "eval_jac_g", "eval_h", "intermediate_callback", "obj_scaling", "x_scaling", "g_scaling", "ipopt_options", NULL},
                                  &n,
                                  &PyArray_Type, &py_x_L,
                                  &PyArray_Type, &py_x_U,
                                  &m,
                                  &PyArray_Type, &py_g_L,
                                  &PyArray_Type, &py_g_U,
                                  &py_sparsity_indices_jac_g,
                                  &py_sparsity_indices_hess,
                                  &callback_data.py_eval_f,
                                  &callback_data.py_eval_grad_f,
                                  &callback_data.py_eval_g,
                                  &callback_data.py_eval_jac_g,
                                  &callback_data.py_eval_h,
                                  &py_intermediate_callback,
                                  &obj_scaling,
                                  &py_x_scaling,
                                  &py_g_scaling,
                                  &py_ipopt_options)
     || !parse_sparsity_indices(py_sparsity_indices_jac_g, &callback_data.sparsity_indices_jac_g)
     || !check_callback(callback_data.py_eval_f, "eval_f")
     || !check_callback(callback_data.py_eval_grad_f, "eval_grad_f")
     || !check_callback(callback_data.py_eval_g, "eval_g")
     || !check_callback(callback_data.py_eval_jac_g, "eval_jac_g")
     || !check_non_negative(m, "m")
     || !check_non_negative(n, "n")
     || !check_array_dim(py_x_L, n, "x_L")
     || !check_array_dim(py_x_U, n, "x_U")
     || !check_array_dim(py_g_L, m, "g_L")
     || !check_array_dim(py_g_U, m, "g_U")
     || !array_copy_data(py_x_L, &x_L)
     || !array_copy_data(py_x_U, &x_U)
     || !array_copy_data(py_g_L, &g_L)
     || !array_copy_data(py_g_U, &g_U)
     || !(callback_data.py_eval_h == NULL || (check_callback(callback_data.py_eval_h, "h") && parse_sparsity_indices(py_sparsity_indices_hess, &callback_data.sparsity_indices_hess)))
     || !check_kwargs(py_ipopt_options)
     || !check_optional(py_x_scaling, _PyArray_Check, "x_scaling", "Optional[numpy.ndarray]")
     || !check_optional(py_g_scaling, _PyArray_Check, "g_scaling", "Optional[numpy.ndarray]")
     ) {
    SAFE_FREE(x_L);
    SAFE_FREE(x_U);
    SAFE_FREE(g_L);
    SAFE_FREE(g_U);
    sparsity_indices_free(&callback_data.sparsity_indices_jac_g);
    sparsity_indices_free(&callback_data.sparsity_indices_hess);
    return NULL;
  }
  
  // Grab the callback selfs because we want to use them later.
  Py_XINCREF(callback_data.py_eval_f);
  Py_XINCREF(callback_data.py_eval_grad_f);
  Py_XINCREF(callback_data.py_eval_g);
  Py_XINCREF(callback_data.py_eval_jac_g);
  Py_XINCREF(callback_data.py_eval_h);

  // create the Ipopt Problem

  self = (IPyOptProblemObject*)type->tp_alloc(type, 0);
  if(!ipopt_problem_c_init(self,
			   n, x_L, x_U,
			   m, g_L, g_U,
			   &callback_data)
     
     || !_set_intermediate_callback((PyObject*)self, py_intermediate_callback)
     || !_set_problem_scaling((PyObject*)self, obj_scaling, (PyArrayObject*)py_x_scaling, (PyArrayObject*)py_g_scaling)) {
    Py_CLEAR(self);
  }
  SAFE_FREE(x_L);
  SAFE_FREE(x_U);
  SAFE_FREE(g_L);
  SAFE_FREE(g_U);
  if(self == NULL) {
    sparsity_indices_free(&callback_data.sparsity_indices_jac_g);
    sparsity_indices_free(&callback_data.sparsity_indices_hess);
  }
  if(!set_options(self->nlp, py_ipopt_options)) {
    Py_XDECREF(self);
    return NULL;
  }
  if(py_ipopt_options != NULL)
    Py_XDECREF(py_ipopt_options);
  return (PyObject*)self;
}

static int py_ipopt_problem_clear(IPyOptProblemObject *self) {
  DispatchData *dp = &self->data;
  
  //Ungrab the callback functions because we do not need them anymore.
  Py_CLEAR(dp->py_eval_f);
  Py_CLEAR(dp->py_eval_grad_f);
  Py_CLEAR(dp->py_eval_g);
  Py_CLEAR(dp->py_eval_jac_g);
  Py_CLEAR(dp->py_eval_h);
  Py_CLEAR(dp->py_intermediate_callback);

  return 0;
}
static void py_ipopt_problem_dealloc(PyObject *self) {
  IPyOptProblemObject *obj = (IPyOptProblemObject*)self;
  DispatchData *dp = &obj->data;

  PyObject_GC_UnTrack(self);
  py_ipopt_problem_clear(obj);

  sparsity_indices_free(&dp->sparsity_indices_jac_g);
  sparsity_indices_free(&dp->sparsity_indices_hess);
  
  FreeIpoptProblem(obj->nlp);
  
  Py_TYPE(self)->tp_free(self);
}

static int py_ipopt_problem_traverse(IPyOptProblemObject *self, visitproc visit, void *arg) {
  DispatchData *dp = &self->data;
  Py_VISIT(dp->py_eval_f);
  Py_VISIT(dp->py_eval_grad_f);
  Py_VISIT(dp->py_eval_g);
  Py_VISIT(dp->py_eval_jac_g);
  Py_VISIT(dp->py_eval_h);
  Py_VISIT(dp->py_intermediate_callback);
  return 0;
}


PyMethodDef problem_methods[] = {
  {"solve", (PyCFunction)solve, METH_VARARGS | METH_KEYWORDS, PyDoc_STR(IPYOPT_SOLVE_DOC)},
  {"set_intermediate_callback", set_intermediate_callback, METH_VARARGS,
   PyDoc_STR(IPYOPT_SET_INTERMEDIATE_CALLBACK_DOC)},
  {"set", (PyCFunction)set, METH_VARARGS | METH_KEYWORDS, PyDoc_STR(IPYOPT_SET_OPTION_DOC)},
  {"set_problem_scaling", (PyCFunction)set_problem_scaling, METH_VARARGS | METH_KEYWORDS, PyDoc_STR(IPYOPT_SET_PROBLEM_SCALING_DOC)},
  {NULL, NULL, 0, NULL},
};

#if PY_MAJOR_VERSION < 3
static PyObject *problem_getattr(PyObject *self, char *attrname) {
  PyObject *result = NULL;
  result = Py_FindMethod(problem_methods, self, attrname);
  return result;
}

/*
 * had to replace PyObject_HEAD_INIT(&PyType_Type) in order to get this to
 * compile on Windows
 */
PyTypeObject IPyOptProblemType = {
  PyObject_HEAD_INIT(NULL)
  .ob_size = 0,
  .tp_name = "ipyopt.Problem",
  .tp_basicsize = sizeof(IPyOptProblemObject),
  .tp_itemsize = 0,
  .tp_dealloc = py_ipopt_problem_dealloc,
  .tp_print = 0,
  .tp_getattr = problem_getattr,
  .tp_setattr = 0,
  .tp_compare = 0,
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
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR(IPYOPT_PROBLEM_DOC),
  .tp_new = py_ipopt_problem_new
};

#else

PyTypeObject IPyOptProblemType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "ipyopt.Problem",
  .tp_basicsize = sizeof(IPyOptProblemObject),
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
  .tp_new = py_ipopt_problem_new
};
#endif

static void sparsity_indices_allocate(SparsityIndices *idx, unsigned int n) {
  idx->row = malloc(n*sizeof(Index));
  idx->col = malloc(n*sizeof(Index));
  idx->n = n;
}
static void sparsity_indices_free(SparsityIndices* idx) {
  if(idx->row != NULL) {
    free(idx->row);
    idx->row = NULL;
  }
  if(idx->col != NULL) {
    free(idx->col);
    idx->col = NULL;
  }
}
static _Bool parse_sparsity_indices(PyObject* obj, SparsityIndices *idx) {
  PyObject *rows, *cols;
  Py_ssize_t n, i;
  if(!PyTuple_Check(obj)) {
    PyErr_Format(PyExc_TypeError, "Sparsity info: a tuple of size 2 is needed.");
    return FALSE;
  }
  if(PyTuple_Size(obj) != 2) {
    PyErr_Format(PyExc_TypeError, "Sparsity info: a tuple of size 2 is needed. Found tuple of size %d", PyTuple_Size(obj));
    return FALSE;
  }
  rows = PyTuple_GetItem(obj, 0);
  cols = PyTuple_GetItem(obj, 1);
  n = PyObject_Length(rows);
  if(n != PyObject_Length(cols)) {
    PyErr_Format(PyExc_TypeError, "Sparsity info: length of row indices (%d) does not match lenth of column indices (%d)",
                 n, PyObject_Length(cols));
    return FALSE;
  }
  sparsity_indices_allocate(idx, n);
  PyObject *row_iter = PyObject_GetIter(rows);
  PyObject *col_iter = PyObject_GetIter(cols);
  PyObject *row_item, *col_item;
  for(i=0; i<n; i++) {
    row_item = PyIter_Next(row_iter);
    col_item = PyIter_Next(col_iter);
    if(row_item != NULL) idx->row[i] = PyLong_AsLong(row_item);
    if(col_item != NULL) idx->col[i] = PyLong_AsLong(col_item);
    if(row_item == NULL || col_item == NULL || PyErr_Occurred() != NULL) {
      PyErr_Format(PyExc_TypeError, "Sparsity info: Row an column indices must be integers");
      sparsity_indices_free(idx);
      return FALSE;
    }
  }
  return TRUE;
}

static Bool check_kwargs(const PyObject *kwargs) {
  if(kwargs == NULL || kwargs == Py_None || PyDict_Check(kwargs)) return TRUE;
  PyErr_Format(PyExc_RuntimeError, "C-API-Level Error: keywords are not of type dict");
  return FALSE;
}
static Bool check_optional(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name) {
  if(obj == NULL || obj == Py_None || checker(obj))
    return TRUE;
  PyErr_Format(PyExc_TypeError, "Wrong type for %s. Required: %s", obj_name, type_name);
  return FALSE;
}
static Bool check_type(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name) {
  if(obj != NULL && check_optional(obj, checker, obj_name, type_name))
    return TRUE;
  PyErr_Format(PyExc_TypeError, "Error while parsing %s.", obj_name);
  return FALSE;
}

// Begin Python Module code section
static PyMethodDef ipoptMethods[] = {
  {NULL, NULL}
};

#if PY_MAJOR_VERSION < 3
typedef struct {
  const char *m_name;
  const char *m_doc;
  Py_ssize_t m_size;
  PyMethodDef* m_methods;
} PyModuleDef;
#define PyModuleDef_HEAD_INIT
#endif

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  .m_name = "ipyopt",
  .m_doc = "A hook between Ipopt and Python",
  .m_size = -1,
  .m_methods = ipoptMethods
};
#if PY_MAJOR_VERSION >= 3
#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF() PyModule_Create(&moduledef)
#else
#define MOD_ERROR_VAL
#define MOD_SUCCESS_VAL(val)
#define MOD_INIT(name) void init##name(void)
#define MOD_DEF() Py_InitModule3(moduledef.name, moduledef.methods, moduledef.doc)
#endif

MOD_INIT(ipyopt) {
  PyObject *module;
  // Finish initialization of the problem type
  if(PyType_Ready(&IPyOptProblemType) < 0) 
    return MOD_ERROR_VAL;
  
  module = MOD_DEF();
    
  if(module == NULL)
    return MOD_ERROR_VAL;

  Py_INCREF(&IPyOptProblemType);
  PyModule_AddObject(module, "Problem", (PyObject*)&IPyOptProblemType);
  
  // Initialize numpy (a segfault will occur if I use numarray without this)
  import_array();
  if(PyErr_Occurred())
    Py_FatalError("Unable to initialize module ipyopt");
  
  return MOD_SUCCESS_VAL(module);
}
