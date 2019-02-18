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
#include "logger.h"

#ifndef SAFE_FREE
#define SAFE_FREE(p) {if(p) {free(p); (p)= NULL;}}
#endif

/*
 * Let's put the static char docs at the beginning of this file...
 */

static char IPYOPT_SOLVE_DOC[] = "solve(x, [mult_g, mult_x_L, mult_x_U]) -> (x, obj, status)"
  "\n\n"
  "Call Ipopt to solve problem created before and return" "\n"
  "a tuple that contains final solution x, final objective function obj," "\n"
  "and the return status of ipopt." "\n\n"
  "mult_g, mult_x_L, mult_x_U are optional keyword only arguments" "\n"
  "allowing previous values of bound multipliers to be passed in warm" "\n"
  "start applications."
  "If passed, these variables are modified.";

static char IPYOPT_SET_INTERMEDIATE_CALLBACK_DOC[] =
  "set_intermediate_callback(callback_function)" "\n\n"
  "Set the intermediate callback function. "
  "This gets called each iteration.";

static char IPYOPT_SET_OPTION_DOC[] =
  "set([key1=val1, ...])" "\n\n"
  "Set one or more Ipopt options. The python type of the value objects have to match "
  "the corresponding types (i.e. str, float or int) of the IPOpt options. "
  "Refer to the Ipopt" "\n"
  "document for more information about Ipopt options, or use" "\n"
  "ipopt --print-options" "\n"
  "to see a list of available options.";

static char IPYOPT_PROBLEM_DOC[] =
  "IPOpt problem type in python" "\n\n"
  "Problem(n, xl, xu, m, gl, gu, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g) -> Problem" "\n\n"
  "n is the number of variables," "\n"
  "xl is the lower bound of x as bounded constraints" "\n"
  "xu is the upper bound of x as bounded constraints" "\n"
  "\t" "both xl, xu should be one dimension arrays with length n" "\n\n"
  "m is the number of constraints," "\n"
  "gl is the lower bound of constraints" "\n"
  "gu is the upper bound of constraints" "\n"
  "\t" "both gl, gu should be one dimension arrays with length m" "\n"
  "nnzj is the number of nonzeros in Jacobi matrix" "\n"
  "nnzh is the number of non-zeros in Hessian matrix, you can set it to 0" "\n\n"
  "eval_f is the call back function to calculate objective value," "\n"
  "it takes one single argument x as input vector" "\n"
  "eval_grad_f calculates gradient for objective function" "\n"
  "eval_g calculates the constraint values and return an array" "\n"
  "eval_jac_g calculates the Jacobi matrix. It takes an argument x" "\n"
  "and returns the values of the Jacobi matrix with length nnzj" "\n"
  "eval_h calculates the hessian matrix, it's optional." "\n"
  "if omitted, please set nnzh to 0 and Ipopt will use approximated hessian" "\n"
  "which will make the convergence slower.";

static char IPYOPT_LOG_DOC[] = "set_loglevel(level)" "\n\n"
  "Set the log level of IPyOpt. All positive integers are allowed. "
  "Messages will be logged if their level is greater or equal than the log level. "
  "However, the log level 0 will turn off logging." "\n"
  "Predefined levels:" "\n"
  "LOGGING_OFF: 0" "\n"
  "LOGGING_INFO: 10" "\n"
  "LOGGING_DEBUG: 10";


static void unpack_args(PyObject *args, PyObject ***unpacked_args, unsigned int *n_args);
static Bool check_type(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name);
static Bool check_type_optional(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name);
static Bool check_callback(PyObject *callback, const char *name);
static Bool _PyArray_Check(const PyObject* obj) { return PyArray_Check(obj); } // Macro -> function
static Bool _PyLong_Check(const PyObject* obj) { return PyLong_Check(obj); } // Macro -> function
static Bool _PyUnicode_Check(const PyObject* obj) { return PyUnicode_Check(obj); } // Macro -> function
static Bool _PyFloat_Check(const PyObject* obj) { return PyFloat_Check(obj); } // Macro -> function
static Bool check_args(const PyObject *args);
static Bool check_kwargs(const PyObject *kwargs);
static Bool check_non_negative(int n, const char *name)
{
  if(n >= 0) return TRUE;
  PyErr_Format(PyExc_ValueError, "%s can't be negative", name);
  return FALSE;
}
static Bool array_copy_data(PyArrayObject *arr, Number **dest)
{
  unsigned int n = PyArray_SIZE(arr), i;
  Number *data = PyArray_DATA(arr);
  *dest = malloc(n * sizeof(Number*));
  if(!*dest)
    {
      PyErr_NoMemory();
      return FALSE;
    }
  for(i=0; i<n; i++)
    (*dest)[i] = data[i];
  return TRUE;
}
static Bool check_array_ndim(PyArrayObject *arr, unsigned int ndim, const char *name)
{
  if((unsigned int)PyArray_NDIM(arr) == ndim) return TRUE;
  PyErr_Format(PyExc_ValueError, "%s has wrong number of dimensions. Expected %d, got %d", name, ndim, PyArray_NDIM(arr));
  return FALSE;
}
static Bool check_array_shape(PyArrayObject *arr, unsigned int dim, const char *name)
{
  if(!check_array_ndim(arr, 1, name)) return FALSE;
  if(PyArray_DIMS(arr)[0] == dim) return TRUE;
  PyErr_Format(PyExc_ValueError, "%s has wrong shape. Expected (%d,), found (%d,)", name, dim, PyArray_DIMS(arr)[0]);
  return FALSE;
}
static Bool check_array_shape_equal(PyArrayObject *arr1, PyArrayObject *arr2, const char *name1, const char *name2)
{
  int n = PyArray_NDIM(arr1), i;
  if(n != PyArray_NDIM(arr2))
    {
      PyErr_Format(PyExc_ValueError, "%s and %s must have the same shape.", name1, name2);
      return FALSE;
    }
  for(i=0; i<n; i++)
    if(PyArray_SHAPE(arr1)[i] != PyArray_SHAPE(arr2)[i])
      {
	PyErr_Format(PyExc_ValueError, "%s and %s must have the same shape.", name1, name2);
	return FALSE;
      }
  return TRUE;
}

typedef struct
{
  PyObject_HEAD IpoptProblem nlp;
  DispatchData data;
  Index n_variables;
  Index m_constraints;
} IPyOptProblemObject;

// Object Section
static PyObject *solve(PyObject *self, PyObject *args, PyObject *keywords);
static PyObject *set_intermediate_callback(PyObject *self, PyObject *args);
static PyObject *py_ipopt_problem_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void py_ipopt_problem_dealloc(PyObject *self);
static int py_ipopt_problem_clear(IPyOptProblemObject *self);
static int py_ipopt_problem_traverse(IPyOptProblemObject *self, visitproc visit, void *arg);


static Bool set_int_option(IpoptProblem nlp, char *key, PyObject *obj)
{ return AddIpoptIntOption(nlp, key, PyLong_AsLong(obj)); }
static Bool set_num_option(IpoptProblem nlp, char *key, PyObject *obj)
{ return AddIpoptNumOption(nlp, key, PyFloat_AsDouble(obj)); }
static Bool set_str_option(IpoptProblem nlp, char *key, PyObject *obj)
{ return AddIpoptStrOption(nlp, key, (char*)PyUnicode_AsUTF8(obj)); }

typedef struct
{
  Bool (*check)(const PyObject*);
  Bool (*set_option)(IpoptProblem, char*, PyObject*);
  char *type_repr;
} type_mapping_record;

const static type_mapping_record type_mapping[] =
  {
   {.check = _PyFloat_Check, .set_option = set_num_option, .type_repr = "num"},
   {.check = _PyUnicode_Check, .set_option = set_str_option, .type_repr = "str"},
   {.check = _PyLong_Check, .set_option = set_int_option, .type_repr = "int"}
  };

static Bool set_option(IpoptProblem nlp, PyObject *key, PyObject *val)
{
  const char *c_key = PyUnicode_AsUTF8(key);
  unsigned int i;
  Bool ret = FALSE;
  for(i=0; i<sizeof(type_mapping)/sizeof(type_mapping_record); i++)
    if((*type_mapping[i].check)(val))
      {
	ret = (*type_mapping[i].set_option)(nlp, (char*)c_key, val);
	if(PyErr_Occurred() != NULL) return FALSE;
	if(!ret)
	  {
	    PyErr_Format(PyExc_ValueError, "%s is not a valid %s option", c_key, type_mapping[i].type_repr);
	    return FALSE;
	  }
	return TRUE;
      }
  PyErr_Format(PyExc_TypeError, "The value for option %s has unsupported type", c_key);
  return FALSE;
}
static Bool set_options(IpoptProblem nlp, PyObject *dict)
{
  PyObject *key, *val;
  Py_ssize_t pos = 0;

  if(dict == NULL) return TRUE;
  while (PyDict_Next(dict, &pos, &key, &val))
    {
      if(!set_option(nlp, key, val))
	return FALSE;
    }
  return TRUE;
}

static Bool check_argument(PyObject *obj, Bool (*check)(PyObject*),
			   void *err, const char *fmt, ...)
{
  if((*check)(obj)) return TRUE;
  va_list ap;
  va_start(ap, fmt);
  PyErr_Format(err, fmt, ap);
  va_end(ap);
  return FALSE;
}
static Bool check_callback(PyObject *callback, const char *name)
{
  return check_argument(callback, PyCallable_Check, PyExc_TypeError, "Need a callable object for callback function %s", name);
}

static Bool check_no_args(const char* f_name, PyObject *args)
{
  if(args == NULL) return TRUE;
  if(!PyTuple_Check(args))
    {
      PyErr_Format(PyExc_RuntimeError, "Argument keywords is not a tuple");
      return FALSE;
    }
  unsigned int n = PyTuple_Size(args);
  if(n == 0) return TRUE;
  PyErr_Format(PyExc_TypeError, "%s() takes 0 positional arguments but %d %s given", f_name, n, n==1?"was":"were");
  return FALSE;
}
static PyObject *set(PyObject *self, PyObject *args, PyObject *keywords)
{
  IpoptProblem nlp = (IpoptProblem)(((IPyOptProblemObject*)self)->nlp);

  if(!check_kwargs(keywords) || !check_no_args("set", args)) return PyErr_Occurred();
  if(!set_options(nlp, keywords))
    return NULL;
  Py_INCREF(Py_None);
  return Py_None;
}


PyMethodDef problem_methods[] =
  {
   {"solve", (PyCFunction)solve, METH_VARARGS | METH_KEYWORDS, PyDoc_STR(IPYOPT_SOLVE_DOC)},
   {"set_intermediate_callback", set_intermediate_callback, METH_VARARGS,
    PyDoc_STR(IPYOPT_SET_INTERMEDIATE_CALLBACK_DOC)},
   {"set", (PyCFunction)set, METH_VARARGS | METH_KEYWORDS, PyDoc_STR(IPYOPT_SET_OPTION_DOC)},
   {NULL, NULL},
  };

#if PY_MAJOR_VERSION < 3
static PyObject *problem_getattr(PyObject *self, char *attrname)
{
  PyObject *result = NULL;
  result = Py_FindMethod(problem_methods, self, attrname);
  return result;
}


/*
 * had to replace PyObject_HEAD_INIT(&PyType_Type) in order to get this to
 * compile on Windows
 */
PyTypeObject IPyOptProblemType =
  {
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

PyTypeObject IPyOptProblemType =
  {
   PyVarObject_HEAD_INIT(NULL, 0)
   .tp_name = "ipyopt.Problem",
   .tp_basicsize = sizeof(IPyOptProblemObject),
   .tp_itemsize = 0,
   .tp_dealloc = (destructor)py_ipopt_problem_dealloc,
   .tp_print = 0,
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


static _Bool ipopt_problem_c_init(IPyOptProblemObject *object,
				  int n, Number *x_L, Number *x_U,
				  int m, Number *g_L, Number *g_U,
				  const DispatchData *callback_data)
{
  int C_indexstyle = 0;
  IpoptProblem thisnlp = CreateIpoptProblem(n,
					    x_L, x_U, m, g_L, g_U,
					    callback_data->sparsity_indices_jac_g.n,
					    callback_data->sparsity_indices_hess.n,
					    C_indexstyle,
					    &eval_f, &eval_g,
					    &eval_grad_f,
					    &eval_jac_g, &eval_h);
  if(thisnlp == NULL)
    {
      PyErr_SetString(PyExc_MemoryError, "Cannot create IpoptProblem instance");
      return FALSE;
    }
  object->n_variables = n;
  object->m_constraints = m;
  object->nlp = thisnlp;
  memcpy((void*)&object->data, (void*)callback_data, sizeof(DispatchData));
  return TRUE;
}

static PyObject *set_loglevel(PyObject *obj, PyObject *args)
{
  int l;
  if(!PyArg_ParseTuple(args, "i", &l) || !logger_set_loglevel(l))
    return NULL;
  Py_INCREF(Py_None);
  return Py_None;
}

static void sparsity_indices_allocate(SparsityIndices *idx, unsigned int n)
{
  idx->row = malloc(n*sizeof(Index));
  idx->col = malloc(n*sizeof(Index));
  idx->n = n;
}
static void sparsity_indices_free(SparsityIndices* idx)
{
  if(idx->row != NULL)
    {
      free(idx->row);
      idx->row = NULL;
    }
  if(idx->col != NULL)
    {
      free(idx->col);
      idx->col = NULL;
    }
}
static _Bool parse_sparsity_indices(PyObject* obj, SparsityIndices *idx)
{
  PyObject *rows, *cols;
  Py_ssize_t n, i;
  if(!PyTuple_Check(obj))
    {
      PyErr_Format(PyExc_TypeError, "Sparsity info: a tuple of size 2 is needed.");
      return FALSE;
    }
  if(PyTuple_Size(obj) != 2)
    {
      PyErr_Format(PyExc_TypeError, "Sparsity info: a tuple of size 2 is needed. Found tuple of size %d", PyTuple_Size(obj));
      return FALSE;
    }
  rows = PyTuple_GetItem(obj, 0);
  cols = PyTuple_GetItem(obj, 1);
  n = PyObject_Length(rows);
  if(n != PyObject_Length(cols))
    {
      PyErr_Format(PyExc_TypeError, "Sparsity info: length of row indices (%d) does not match lenth of column indices (%d)",
		   n, PyObject_Length(cols));
      return FALSE;
    }
  sparsity_indices_allocate(idx, n);
  PyObject *row_iter = PyObject_GetIter(rows);
  PyObject *col_iter = PyObject_GetIter(cols);
  PyObject *row_item, *col_item;
  for(i=0; i<n; i++)
    {
      row_item = PyIter_Next(row_iter);
      col_item = PyIter_Next(col_iter);
      if(row_item != NULL) idx->row[i] = PyLong_AsLong(row_item);
      if(col_item != NULL) idx->col[i] = PyLong_AsLong(col_item);
      if(row_item == NULL || col_item == NULL || PyErr_Occurred() != NULL)
	{
	  PyErr_Format(PyExc_TypeError, "Sparsity info: Row an column indices must be integers");
	  sparsity_indices_free(idx);
	  return FALSE;
	}
    }
  return TRUE;
}

static PyObject *py_ipopt_problem_new(PyTypeObject *type, PyObject *args, PyObject *keywords)
{
  IPyOptProblemObject *self = NULL;

  DispatchData callback_data = {
				.eval_f_python = NULL,
				.eval_grad_f_python = NULL,
				.eval_g_python = NULL,
				.eval_jac_g_python = NULL,
				.eval_h_python = NULL,
				.apply_new_python = NULL,
				.callback_args = NULL,
				.n_callback_args = 0,
				.callback_kwargs = NULL,
				.sparsity_indices_jac_g = { 0 },
				.sparsity_indices_hess = { 0 }
  };
  int n;			// Number of variables
  PyArrayObject *xL = NULL;
  PyArrayObject *xU = NULL;
  int m;			// Number of constraints
  PyArrayObject *gL = NULL;
  PyArrayObject *gU = NULL;
  
  Number *x_L = NULL;	// lower bounds on x
  Number *x_U = NULL;	// upper bounds on x
  Number *g_L = NULL;	// lower bounds on g
  Number *g_U = NULL;	// upper bounds on g
  
  PyObject *sparsity_indices_jac_g = NULL;
  PyObject *sparsity_indices_hess = NULL;
  
  // Init the callback_data field
  if(!PyArg_ParseTuple(args, "iO!O!iO!O!OOOOOO|OOO:ipyoptcreate",
		       &n,
		       &PyArray_Type, &xL,
		       &PyArray_Type, &xU,
		       &m,
		       &PyArray_Type, &gL,
		       &PyArray_Type, &gU,
		       &sparsity_indices_jac_g,
		       &sparsity_indices_hess,
		       &callback_data.eval_f_python,
		       &callback_data.eval_grad_f_python,
		       &callback_data.eval_g_python,
		       &callback_data.eval_jac_g_python,
		       &callback_data.eval_h_python,
		       &callback_data.apply_new_python)
     || !parse_sparsity_indices(sparsity_indices_jac_g, &callback_data.sparsity_indices_jac_g)
     || !check_kwargs(keywords)
     || !check_callback(callback_data.eval_f_python, "eval_f")
     || !check_callback(callback_data.eval_grad_f_python, "eval_grad_f")
     || !check_callback(callback_data.eval_g_python, "eval_g")
     || !check_callback(callback_data.eval_jac_g_python, "eval_jac_g")
     || (callback_data.apply_new_python != NULL && !check_callback(callback_data.apply_new_python, "applynew"))
     || !check_non_negative(m, "m")
     || !check_non_negative(n, "n")
     || !check_array_shape(xL, n, "x_L")
     || !check_array_shape(xU, n, "x_U")
     || !check_array_shape(gL, m, "g_L")
     || !check_array_shape(gU, m, "g_U")
     || !array_copy_data(xL, &x_L)
     || !array_copy_data(xU, &x_U)
     || !array_copy_data(gL, &g_L)
     || !array_copy_data(gU, &g_U)
     || !(callback_data.eval_h_python == NULL || (check_callback(callback_data.eval_h_python, "h") && parse_sparsity_indices(sparsity_indices_hess, &callback_data.sparsity_indices_hess))) 
     )
    {
      SAFE_FREE(x_L);
      SAFE_FREE(x_U);
      SAFE_FREE(g_L);
      SAFE_FREE(g_U);
      sparsity_indices_free(&callback_data.sparsity_indices_jac_g);
      sparsity_indices_free(&callback_data.sparsity_indices_hess);
      return NULL;
    }
  if(callback_data.eval_h_python == NULL)
    logger(LOG_INFO, "Ipopt will use Hessian approximation.\n");
  
  // Grab the callback selfs because we want to use them later.
  Py_XINCREF(callback_data.eval_f_python);
  Py_XINCREF(callback_data.eval_grad_f_python);
  Py_XINCREF(callback_data.eval_g_python);
  Py_XINCREF(callback_data.eval_jac_g_python);
  Py_XINCREF(callback_data.eval_h_python);
  Py_XINCREF(callback_data.apply_new_python);

  // create the Ipopt Problem

  self = (IPyOptProblemObject*)type->tp_alloc(type, 0);
  if(!ipopt_problem_c_init(self,
			   n, x_L, x_U,
			   m, g_L, g_U,
			   &callback_data))
    {
      Py_CLEAR(self);
    }
  SAFE_FREE(x_L);
  SAFE_FREE(x_U);
  SAFE_FREE(g_L);
  SAFE_FREE(g_U);
  if(self == NULL)
    {
      sparsity_indices_free(&callback_data.sparsity_indices_jac_g);
      sparsity_indices_free(&callback_data.sparsity_indices_hess);
    }
  if(!set_options(self->nlp, keywords))
    {
      Py_XDECREF(self);
      return NULL;
    }
  logger(LOG_FULL, "Problem created");
  return (PyObject*)self;
}

static PyObject *set_intermediate_callback(PyObject *self, PyObject *args)
{
  PyObject *intermediate_callback;
  IPyOptProblemObject *temp = (IPyOptProblemObject*)self;
  IpoptProblem nlp = temp->nlp;
  DispatchData *bigfield = (DispatchData*)&temp->data;
  
  if(!PyArg_ParseTuple(args, "O", &intermediate_callback)
     || !check_callback(intermediate_callback, "intermediate_callback"))
    return NULL;
  bigfield->eval_intermediate_callback_python = intermediate_callback;
      
  // Put a Python function object into this data structure
  SetIntermediateCallback(nlp, eval_intermediate_callback);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *solve(PyObject *self, PyObject *args, PyObject *keywords)
{
  enum ApplicationReturnStatus status;	// Solve return code
  int i;
  int n;
  
  // Return values
  IPyOptProblemObject *temp = (IPyOptProblemObject*)self;
  
  IpoptProblem nlp = temp->nlp;
  if(nlp == NULL)
    {
      PyErr_SetString(PyExc_RuntimeError, "nlp objective passed to solve is NULL\n Problem created?\n");
      return NULL;
    }
  DispatchData *bigfield = (DispatchData*)&temp->data;
  int m = temp->m_constraints;
  
  npy_intp dX[1];
  
  PyArrayObject *x = NULL, *mL = NULL, *mU = NULL, *lambda = NULL;
  double *mL_data=NULL, *mU_data=NULL, *lambda_data=NULL;
  PyObject *callback_args = NULL, *callback_kwargs = NULL;
  Number obj; // objective value
  
  PyObject *retval = NULL;
  PyArrayObject *x0 = NULL;
  
  Number *x_working = NULL;
  
  unsigned int n_args = 0;
  PyObject **unpacked_args = NULL;
  if(!PyArg_ParseTupleAndKeywords(args, keywords, "O!|OO$O!O!O!",
				  (char*[]){"x0", "callback_args", "callback_kwargs", "mult_g", "mult_x_L", "mult_x_U", NULL},
				  &PyArray_Type, &x0,
				  &callback_args,
				  &callback_kwargs,
				  &PyArray_Type, &lambda, // mult_g 
				  &PyArray_Type, &mL, // mult_x_L
				  &PyArray_Type, &mU) // mult_x_Y
     || !check_type((PyObject*)x0, &_PyArray_Check, "x0", "numpy.ndarray")
     || !check_type_optional(callback_kwargs, &check_kwargs, "callback_kwargs", "dict")
     || !check_type_optional(callback_args, &check_args, "callback_args", "tuple")
     || !check_array_ndim(x0, 1, "x0")
     || (mL && !check_array_shape_equal(mL, x0, "mL", "x0"))
     || (mU && !check_array_shape_equal(mU, x0, "mU", "x0"))
     || (lambda && !check_array_shape(lambda, m, "lambda"))
     || !array_copy_data(x0, &x_working)
     )
    {
      SAFE_FREE(x_working);
      return NULL;
    }
  if(callback_args != Py_None && callback_args != NULL)
    unpack_args(callback_args, &unpacked_args, &n_args);
  if(callback_kwargs == Py_None) callback_kwargs = NULL;
  
  bigfield->callback_args = unpacked_args;
  bigfield->n_callback_args = n_args;
  bigfield->callback_kwargs = callback_kwargs;
  if(bigfield->eval_h_python == NULL)
    AddIpoptStrOption(nlp, "hessian_approximation", "limited-memory");
  
  // allocate space for the initial point and set the values
  n = PyArray_DIMS(x0)[0];
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
  if(x != NULL && mL_data != NULL && mU_data != NULL && lambda_data !=NULL)
    {
      status = IpoptSolve(nlp, x_working, NULL, &obj,
			  lambda_data, mL_data, mU_data,
			  (UserDataPtr)bigfield);
      double *return_x_data = PyArray_DATA(x);
      for(i=0; i<n; i++)
	return_x_data[i] = x_working[i];

      retval = PyTuple_Pack(3, PyArray_Return(x), PyFloat_FromDouble(obj), PyLong_FromLong(status));
    }
  else
    {
      Py_XDECREF(x);
      PyErr_NoMemory();
    }
  // clean up and return
  if(lambda == NULL && lambda_data != NULL) free(lambda_data);
  if(mU == NULL && mU_data != NULL) free(mU_data);
  if(mL == NULL && mL_data != NULL) free(mL_data);

  Py_XDECREF(x);
  SAFE_FREE(x_working);
  SAFE_FREE(unpacked_args);
  return retval;
}
static void unpack_args(PyObject *args, PyObject ***unpacked_args, unsigned int *n_args)
{
  unsigned int i, n;
  n = PyTuple_Size(args);
  *n_args = n;
  if(n == 0)
    *unpacked_args = NULL;
  else
    *unpacked_args = malloc(n*sizeof(PyObject*));
  for(i=0; i<n; i++)
    (*unpacked_args)[i] = PyTuple_GetItem(args, i);
}
static Bool check_args(const PyObject *args)
{
  return !(args != NULL && args != Py_None && !PyTuple_Check(args));
}
static Bool check_kwargs(const PyObject *kwargs)
{
  if(kwargs == NULL || kwargs == Py_None || PyDict_Check(kwargs)) return TRUE;
  PyErr_Format(PyExc_RuntimeError, "C-API-Level Error: keywords are not of type dict");
  return FALSE;
}
static Bool check_type_optional(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name)
{
  if(obj == NULL || checker(obj))
    return TRUE;
  PyErr_Format(PyExc_TypeError, "Wrong type for %s. Required: %s", obj_name, type_name);
  return FALSE;
}
static Bool check_type(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name)
{
  if(obj != NULL && check_type_optional(obj, checker, obj_name, type_name))
    return TRUE;
  PyErr_Format(PyExc_TypeError, "Error while parsing %s.", obj_name);
  return FALSE;
}
static int py_ipopt_problem_clear(IPyOptProblemObject *self)
{
  DispatchData *dp = &self->data;
  
  //Ungrab the callback functions because we do not need them anymore.
  Py_CLEAR(dp->eval_f_python);
  Py_CLEAR(dp->eval_grad_f_python);
  Py_CLEAR(dp->eval_g_python);
  Py_CLEAR(dp->eval_jac_g_python);
  Py_CLEAR(dp->eval_h_python);
  Py_CLEAR(dp->apply_new_python);

  return 0;
}
static int py_ipopt_problem_traverse(IPyOptProblemObject *self, visitproc visit, void *arg)
{
  DispatchData *dp = &self->data;
  Py_VISIT(dp->eval_f_python);
  Py_VISIT(dp->eval_grad_f_python);
  Py_VISIT(dp->eval_g_python);
  Py_VISIT(dp->eval_jac_g_python);
  Py_VISIT(dp->eval_h_python);
  Py_VISIT(dp->apply_new_python);
  return 0;
}

static void py_ipopt_problem_dealloc(PyObject *self)
{
  IPyOptProblemObject *obj = (IPyOptProblemObject*)self;
  DispatchData *dp = &obj->data;

  PyObject_GC_UnTrack(self);
  py_ipopt_problem_clear(obj);

  sparsity_indices_free(&dp->sparsity_indices_jac_g);
  sparsity_indices_free(&dp->sparsity_indices_hess);
  
  FreeIpoptProblem(obj->nlp);
  
  Py_TYPE(self)->tp_free(self);
  logger(LOG_FULL, "Problem deallocated");
}

// Begin Python Module code section
static PyMethodDef ipoptMethods[] =
  {
   {"set_loglevel", set_loglevel, METH_VARARGS, PyDoc_STR(IPYOPT_LOG_DOC)},
   {NULL, NULL}
  };

#if PY_MAJOR_VERSION < 3
typedef struct
{
  const char *m_name;
  const char *m_doc;
  Py_ssize_t m_size;
  PyMethodDef* m_methods;
} PyModuleDef;
#define PyModuleDef_HEAD_INIT
#endif

static struct PyModuleDef moduledef =					
  {
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

MOD_INIT(ipyopt)
{
  PyObject *module;
  // Finish initialization of the problem type
  if(PyType_Ready(&IPyOptProblemType) < 0) 
    return MOD_ERROR_VAL;
  
  module = MOD_DEF();
    
  if(module == NULL)
    return MOD_ERROR_VAL;

  Py_INCREF(&IPyOptProblemType);
  PyModule_AddObject(module, "Problem", (PyObject*)&IPyOptProblemType);
  
  logger_register_log_levels(module);
  
  // Initialize numpy (a segfault will occur if I use numarray without this)
  import_array();
  if(PyErr_Occurred())
    Py_FatalError("Unable to initialize module ipyopt");
  
  return MOD_SUCCESS_VAL(module);
}
