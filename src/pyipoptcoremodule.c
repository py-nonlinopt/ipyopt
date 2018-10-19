/*  Author: Eric Xu                                       */
/*  Licensed under BSD                                    */
/*                                                        */
/*  Modifications on logger made by                       */
/*  OpenMDAO at NASA Glenn Research Center, 2010 and 2011 */
/*  Modifications on the SAFE_FREE macro made by          */
/*  Guillaume Jacquenot, 2012                             */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL pyipopt_ARRAY_API
#include "numpy/arrayobject.h"
#include "hook.h"

#ifndef SAFE_FREE
#define SAFE_FREE(p) {if(p) {free(p); (p)= NULL;}}
#endif

/*
 * Let's put the static char docs at the beginning of this file...
 */

static char PYIPOPT_SOLVE_DOC[] = "solve(x, [mult_g, mult_x_L, mult_x_U]) -> (x, obj, status)\n \
  \n									\
  Call Ipopt to solve problem created before and return  \n \
  a tuple that contains final solution x, final objective function obj, \n \
  and the return status of ipopt. \n \
  \n \
  mult_g, mult_x_L, mult_x_U are optional keyword only arguments \n \
  allowing previous values of bound multipliers to be passed in warm \n \
  start applications. \
  If passed, these variables are modified.";

static char PYIPOPT_SET_INTERMEDIATE_CALLBACK_DOC[] =
  "set_intermediate_callback(callback_function)\n \
  \n                                              \
  Set the intermediate callback function.         \
  This gets called each iteration.";

static char PYIPOPT_CLOSE_DOC[] = "After all the solving, close the model\n";

static char PYIPOPT_SET_OPTION_DOC[] =
  "set([key1=val1, ...])\n\n \
   Set one or more Ipopt options. Refer to the Ipopt \n		       \
     document for more information about Ipopt options, or use \n      \
       ipopt --print-options \n					       \
     to see a list of available options.";

static char PYIPOPT_CREATE_DOC[] =
  "create(n, xl, xu, m, gl, gu, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g) -> Boolean\n \
       \n \
       Create a problem instance and return True if succeed  \n \
       \n \
       n is the number of variables, \n \
       xl is the lower bound of x as bounded constraints \n \
       xu is the upper bound of x as bounded constraints \n \
               both xl, xu should be one dimension arrays with length n \n \
       \n \
       m is the number of constraints, \n \
       gl is the lower bound of constraints \n \
       gu is the upper bound of constraints \n \
               both gl, gu should be one dimension arrays with length m \n \
       nnzj is the number of nonzeros in Jacobi matrix \n \
       nnzh is the number of non-zeros in Hessian matrix, you can set it to 0 \n \
       \n \
       eval_f is the call back function to calculate objective value, \n \
               it takes one single argument x as input vector \n \
       eval_grad_f calculates gradient for objective function \n \
       eval_g calculates the constraint values and return an array\n \
       eval_jac_g calculates the Jacobi matrix. It takes an argument x\n \
               and returns the values of the Jacobi matrix with length nnzj \n \
       eval_h calculates the hessian matrix, it's optional. \n \
               if omitted, please set nnzh to 0 and Ipopt will use approximated hessian \n \
               which will make the convergence slower. ";

static char PYIPOPT_LOG_DOC[] = "set_loglevel(level)\n \
    \n \
    Set the log level of PyIPOPT \n \
    levels: \n \
        0:  Terse,    no log from pyipopt \n \
        1:  Moderate, logs for ipopt \n \
        2:  Verbose,  logs for both ipopt and pyipopt. \n";



int user_log_level = TERSE;

static void unpack_args(PyObject *args, PyObject ***unpacked_args, unsigned int *n_args);
Bool check_type(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name);
Bool check_type_optional(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name);
static Bool _PyArray_Check(const PyObject* obj) { return PyArray_Check(obj); } // Macro -> function
static Bool check_args(const PyObject *args);
static Bool check_kwargs(const PyObject *kwargs);

// Object Section
// sig of this is void foo(PyO*)
static void problem_dealloc(PyObject *self)
{
  problem *temp = (problem*)self;
  SAFE_FREE(temp->data);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *solve(PyObject *self, PyObject *args, PyObject *keywords);
PyObject *set_intermediate_callback(PyObject *self, PyObject *args);
PyObject *close_model(PyObject *self, PyObject *args);

static Bool _PyLong_Check(const PyObject* obj) { return PyLong_Check(obj); } // Macro -> function
static Bool _PyUnicode_Check(const PyObject* obj) { return PyUnicode_Check(obj); } // Macro -> function
static Bool _PyFloat_Check(const PyObject* obj) { return PyFloat_Check(obj); } // Macro -> function

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

static PyObject* add_option(IpoptProblem nlp, PyObject *key, PyObject *val)
{
  const char *c_key = PyUnicode_AsUTF8(key);
  unsigned int i;
  Bool ret = FALSE;
  for(i=0; i<sizeof(type_mapping)/sizeof(type_mapping_record); i++)
    if((*type_mapping[i].check)(val))
      {
	ret = (*type_mapping[i].set_option)(nlp, (char*)c_key, val);
	if(!ret || PyErr_Occurred() != NULL)
	  return PyErr_Format(PyExc_ValueError, "%s is not a valid %s option", c_key, type_mapping[i].type_repr);
	return NULL;
      }
  return PyErr_Format(PyExc_TypeError, "The value for option %s has unsupported type", c_key);
}
static PyObject* add_options(IpoptProblem nlp, PyObject *dict)
{
  PyObject *key, *val;
  Py_ssize_t pos = 0;
  PyObject *err;

  if(dict == NULL) return NULL;
  while (PyDict_Next(dict, &pos, &key, &val))
    {
      err = add_option(nlp, key, val);
      if(err != NULL)
	return err;
    }
  return NULL;
}

static Bool check_no_args(const char* f_name, PyObject *args)
{
  if(args == NULL) return TRUE;
  if(!PyTuple_Check(args))
    {
      PyErr_Format(PyExc_RuntimeError, "Argument keywords is not a dict");
      return FALSE;
    }
  unsigned int n = PyTuple_Size(args);
  if(n == 0) return TRUE;
  PyErr_Format(PyExc_TypeError, "%s() takes 0 positional arguments but %d %s given", f_name, n, n==1?"was":"were");
  return FALSE;
}
static PyObject *set(PyObject *self, PyObject *args, PyObject *keywords)
{
  IpoptProblem nlp = (IpoptProblem)(((problem*)self)->nlp);

  PyObject *err;
  if(!check_kwargs(keywords) || !check_no_args("set", args)) return PyErr_Occurred();
  err = add_options(nlp, keywords);
  if(err)
    return err;
  return Py_True;
}


PyMethodDef problem_methods[] =
  {
   {"solve", (PyCFunction)solve, METH_VARARGS | METH_KEYWORDS, PYIPOPT_SOLVE_DOC},
   {"set_intermediate_callback", set_intermediate_callback, METH_VARARGS,
    PYIPOPT_SET_INTERMEDIATE_CALLBACK_DOC},
   {"close", close_model, METH_VARARGS, PYIPOPT_CLOSE_DOC},
   {"set", (PyCFunction)set, METH_VARARGS | METH_KEYWORDS, PYIPOPT_SET_OPTION_DOC},
   {NULL, NULL},
  };

#if PY_MAJOR_VERSION < 3
PyObject *problem_getattr(PyObject *self, char *attrname)
{
  PyObject *result = NULL;
  result = Py_FindMethod(problem_methods, self, attrname);
  return result;
}


/*
 * had to replace PyObject_HEAD_INIT(&PyType_Type) in order to get this to
 * compile on Windows
 */
PyTypeObject IpoptProblemType =
  {
   PyObject_HEAD_INIT(NULL)
   0,			// ob_size
   "pyipoptcore.Problem",	// tp_name
   sizeof(problem),	// tp_basicsize
   0,			// tp_itemsize
   problem_dealloc,	// tp_dealloc
   0,			// tp_print
   problem_getattr,	// tp_getattr
   0,			// tp_setattr
   0,			// tp_compare
   0,			// tp_repr
   0,			// tp_as_number
   0,			// tp_as_sequence
   0,			// tp_as_mapping
   0,			// tp_hash
   0,			// tp_call
   0,			// tp_str
   0,			// tp_getattro
   0,			// tp_setattro
   0,			// tp_as_buffer
   Py_TPFLAGS_DEFAULT,	// tp_flags
   "The IPOPT problem object in python",	// tp_doc
  };

#else

PyDoc_STRVAR(IpoptProblemType__doc__, "The IPOPT problem object in python");

PyTypeObject IpoptProblemType =
  {
   PyVarObject_HEAD_INIT(NULL, 0)
   "pyipoptcore.Problem",	// tp_name
   sizeof(problem),         //tp_basicsize*/
   0,                       //tp_itemsize*/
   // methods
   (destructor)problem_dealloc,   //tp_dealloc*/
   (printfunc)0,           //tp_print*/
   0,                      //tp_getattr*/
   0,                      //tp_setattr*/
   0,                      //tp_reserved*/
   (reprfunc)0,            //tp_repr*/
   0,                      //tp_as_number*/
   0,                      //tp_as_sequence*/
   0,                      //tp_as_mapping*/
   (hashfunc)0,            //tp_hash*/
   (ternaryfunc)0,         //tp_call*/
   (reprfunc)0,            //tp_str*/
   (getattrofunc)0,        // tp_getattro
   (setattrofunc)0,        // tp_setattro
   0,                      // tp_as_buffer
   Py_TPFLAGS_DEFAULT,     //tp_flags*/
   IpoptProblemType__doc__,    // tp_doc - Documentation string
   (traverseproc)0,        // tp_traverse
   (inquiry)0,                // tp_clear
   0,                              // tp_richcompare
   0,                              // tp_weaklistoffset
   0,                              // tp_iter
   0,                              // tp_iternext
   problem_methods,               // tp_methods
  };
#endif

/*
 * FIXME: use module or package constants for the log levels,
 * either in pyipoptcore or in the parent package.
 * They are currently #defined in a header file.
 */
static PyObject *set_loglevel(PyObject *obj, PyObject *args)
{
  int l;
  if(!PyArg_ParseTuple(args, "i", &l))
    {
      PySys_WriteStdout("l is %d \n", l);
      return NULL;
    }
  if(l < 0 || l > 2) return NULL;
  user_log_level = l;
  Py_INCREF(Py_True);
  return Py_True;
}

static void sparsity_indices_allocate(SparsityIndices *idx, unsigned int n)
{
  idx->row = malloc(n*sizeof(Index));
  idx->col = malloc(n*sizeof(Index));
  idx->n = n;
}
static void sparsity_indices_free(SparsityIndices* idx)
{
  if(idx->row != NULL) free(idx->row);
  if(idx->col != NULL) free(idx->col);
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


static PyObject *create(PyObject *obj, PyObject *args, PyObject *keywords)
{
  PyObject *applynew = NULL;
  
  DispatchData myowndata = {
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

  /*
   * I have to create a new python object here, return this python object
   */
  
  int n;			// Number of variables
  PyArrayObject *xL = NULL;
  PyArrayObject *xU = NULL;
  int m;			// Number of constraints
  PyArrayObject *gL = NULL;
  PyArrayObject *gU = NULL;
  
  problem *object = NULL;
  
  Number *x_L = NULL;	// lower bounds on x
  Number *x_U = NULL;	// upper bounds on x
  Number *g_L = NULL;	// lower bounds on g
  Number *g_U = NULL;	// upper bounds on g
  
  double *xldata, *xudata;
  double *gldata, *gudata;
  
  int i;
  
  DispatchData *dp = NULL;
  
  PyObject *sparsity_indices_jac_g = NULL;
  PyObject *sparsity_indices_hess = NULL;
  
  // Init the myowndata field
  if(!PyArg_ParseTuple(args, "iO!O!iO!O!OOOOOO|OOO:pyipoptcreate",
		       &n,
		       &PyArray_Type, &xL,
		       &PyArray_Type, &xU,
		       &m,
		       &PyArray_Type, &gL,
		       &PyArray_Type, &gU,
		       &sparsity_indices_jac_g,
		       &sparsity_indices_hess,
		       &myowndata.eval_f_python,
		       &myowndata.eval_grad_f_python,
		       &myowndata.eval_g_python,
		       &myowndata.eval_jac_g_python,
		       &myowndata.eval_h_python,
		       &applynew)
     || !parse_sparsity_indices(sparsity_indices_jac_g, &myowndata.sparsity_indices_jac_g)
     || !check_kwargs(keywords))
    {
      SAFE_FREE(x_L);
      SAFE_FREE(x_U);
      SAFE_FREE(g_L);
      SAFE_FREE(g_U);
      return NULL;
    }
  if(!PyCallable_Check(myowndata.eval_f_python)
     || !PyCallable_Check(myowndata.eval_grad_f_python)
     || !PyCallable_Check(myowndata.eval_g_python)
     || !PyCallable_Check(myowndata.eval_jac_g_python))
    {
      PyErr_SetString(PyExc_TypeError, "Need a callable object for callback functions");
      SAFE_FREE(x_L);
      SAFE_FREE(x_U);
      SAFE_FREE(g_L);
      SAFE_FREE(g_U);
      sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
      sparsity_indices_free(&myowndata.sparsity_indices_hess);
      return NULL;
    }
  if(myowndata.eval_h_python != NULL)
    {
      if(!PyCallable_Check(myowndata.eval_h_python))
	{
	  PyErr_SetString(PyExc_TypeError, "Need a callable object for function h.");
	  SAFE_FREE(x_L);
	  SAFE_FREE(x_U);
	  SAFE_FREE(g_L);
	  SAFE_FREE(g_U);
	  sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
	  sparsity_indices_free(&myowndata.sparsity_indices_hess);
	  return NULL;
	}
      if(!parse_sparsity_indices(sparsity_indices_hess, &myowndata.sparsity_indices_hess))
	{
	  SAFE_FREE(x_L);
	  SAFE_FREE(x_U);
	  SAFE_FREE(g_L);
	  SAFE_FREE(g_U);
	  sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
	  sparsity_indices_free(&myowndata.sparsity_indices_hess);
	  return NULL;
	}
  }
  else logger("[PyIPOPT] Ipopt will use Hessian approximation.\n");
  
  if(applynew != NULL)
    {
      if(PyCallable_Check(applynew))
	  myowndata.apply_new_python = applynew;
      else
	{
	  PyErr_SetString(PyExc_TypeError, "Need a callable object for function applynew.");
	  SAFE_FREE(x_L);
	  SAFE_FREE(x_U);
	  SAFE_FREE(g_L);
	  SAFE_FREE(g_U);
	  sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
	  sparsity_indices_free(&myowndata.sparsity_indices_hess);
	  return NULL;
	}
    }
  if(m < 0 || n < 0)
    {
      PyErr_SetString(PyExc_TypeError, "m or n can't be negative");
      SAFE_FREE(x_L);
      SAFE_FREE(x_U);
      SAFE_FREE(g_L);
      SAFE_FREE(g_U);
      sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
      sparsity_indices_free(&myowndata.sparsity_indices_hess);
      return NULL;
    }
  x_L = (Number*) malloc(sizeof(Number) * n);
  x_U = (Number*) malloc(sizeof(Number) * n);
  if(!x_L || !x_U)
    {
      SAFE_FREE(x_L);
      SAFE_FREE(x_U);
      SAFE_FREE(g_L);
      SAFE_FREE(g_U);
      sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
      sparsity_indices_free(&myowndata.sparsity_indices_hess);
      return PyErr_NoMemory();
    }
  xldata = PyArray_DATA(xL);
  xudata = PyArray_DATA(xU);
  for(i=0; i<n; i++)
    {
      x_L[i] = xldata[i];
      x_U[i] = xudata[i];
    }

  g_L = (Number*) malloc(sizeof(Number) * m);
  g_U = (Number*) malloc(sizeof(Number) * m);
  if(!g_L || !g_U) PyErr_NoMemory();
  
  gldata = PyArray_DATA(gL);
  gudata = PyArray_DATA(gU);
  
  for(i=0; i<m; i++)
    {
      g_L[i] = gldata[i];
      g_U[i] = gudata[i];
    }

  // Grab the callback objects because we want to use them later.
  Py_XINCREF(myowndata.eval_f_python);
  Py_XINCREF(myowndata.eval_grad_f_python);
  Py_XINCREF(myowndata.eval_g_python);
  Py_XINCREF(myowndata.eval_jac_g_python);
  Py_XINCREF(myowndata.eval_h_python);
  Py_XINCREF(applynew);

  // create the Ipopt Problem

  int C_indexstyle = 0;
  IpoptProblem thisnlp = CreateIpoptProblem(n,
					    x_L, x_U, m, g_L, g_U,
					    myowndata.sparsity_indices_jac_g.n,
					    myowndata.sparsity_indices_hess.n,
					    C_indexstyle,
					    &eval_f, &eval_g,
					    &eval_grad_f,
					    &eval_jac_g, &eval_h);
  logger("[PyIPOPT] Problem created");
  if(!thisnlp)
    {
      PyErr_SetString(PyExc_MemoryError, "Cannot create IpoptProblem instance");
      SAFE_FREE(x_L);
      SAFE_FREE(x_U);
      SAFE_FREE(g_L);
      SAFE_FREE(g_U);
      sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
      sparsity_indices_free(&myowndata.sparsity_indices_hess);
      return NULL;
    }
  object = PyObject_NEW(problem, &IpoptProblemType);
  
  if(object != NULL)
    {
      object->n_variables = n;
      object->m_constraints = m;
      object->nlp = thisnlp;
      dp = (DispatchData*) malloc(sizeof(DispatchData));
      if(!dp)
	{
	  SAFE_FREE(x_L);
	  SAFE_FREE(x_U);
	  SAFE_FREE(g_L);
	  SAFE_FREE(g_U);
	  sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
	  sparsity_indices_free(&myowndata.sparsity_indices_hess);
	  return PyErr_NoMemory();
	}
      memcpy((void*)dp, (void*)&myowndata, sizeof(DispatchData));
      object->data = dp;
      SAFE_FREE(x_L);
      SAFE_FREE(x_U);
      SAFE_FREE(g_L);
      SAFE_FREE(g_U);
      PyObject *err = add_options(thisnlp, keywords);
      if(err != NULL) return err;
      return (PyObject*)object;
    }
  else
    {
      PyErr_SetString(PyExc_MemoryError, "Can't create a new Problem instance");
      SAFE_FREE(x_L);
      SAFE_FREE(x_U);
      SAFE_FREE(g_L);
      SAFE_FREE(g_U);
      sparsity_indices_free(&myowndata.sparsity_indices_jac_g);
      sparsity_indices_free(&myowndata.sparsity_indices_hess);
      return NULL;
    }
}

PyObject *set_intermediate_callback(PyObject *self, PyObject *args)
{
  PyObject *intermediate_callback;
  problem *temp = (problem*) self;
  IpoptProblem nlp = (IpoptProblem) (temp->nlp);
  DispatchData *bigfield = (DispatchData*) (temp->data);
  
  if(!PyArg_ParseTuple(args, "O", &intermediate_callback)) return NULL;
  if(!PyCallable_Check(intermediate_callback))
    {
      PyErr_SetString(PyExc_TypeError, "Need a callable object for function!");
      return NULL;
    }
  else
    {
      
      bigfield->eval_intermediate_callback_python =
	intermediate_callback;
      
      // Put a Python function object into this data structure
      SetIntermediateCallback(nlp, eval_intermediate_callback);
      Py_INCREF(Py_True);
      return Py_True;
    }
}



#define SOLVE_CLEANUP() \
  {						\
    if(retval == NULL) {			\
      Py_XDECREF(x);				\
      Py_XDECREF(mL);				\
      Py_XDECREF(mU);				\
      Py_XDECREF(lambda);			\
    }						\
    SAFE_FREE(newx0);				\
    SAFE_FREE(unpacked_args);			\
    return retval;				\
  }

#define SOLVE_CLEANUP_NULL() \
  {			       \
    retval = NULL;	       \
    SOLVE_CLEANUP();	       \
  }

#define SOLVE_CLEANUP_MEMORY()	 \
  {				 \
    retval = PyErr_NoMemory();	 \
    SOLVE_CLEANUP();		 \
  }

#define SOLVE_CLEANUP_TYPE(err) \
  {					   \
    PyErr_SetString(PyExc_TypeError, err); \
    retval = NULL;			   \
    SOLVE_CLEANUP();			   \
  }

PyObject *solve(PyObject *self, PyObject *args, PyObject *keywords)
{
  enum ApplicationReturnStatus status;	// Solve return code
  int i;
  int n;
  
  // Return values
  problem *temp = (problem*) self;
  
  IpoptProblem nlp = (IpoptProblem) (temp->nlp);
  DispatchData *bigfield = (DispatchData*) (temp->data);
  int m = temp->m_constraints;
  
  npy_intp dX[1];
  
  PyArrayObject *x = NULL, *mL = NULL, *mU = NULL, *lambda = NULL;
  double *mL_data=NULL, *mU_data=NULL, *lambda_data=NULL;
  PyObject *callback_args = NULL, *callback_kwargs = NULL;
  Number obj; // objective value
  
  PyObject *retval = NULL;
  PyArrayObject *x0 = NULL;
  
  Number *newx0 = NULL;
  
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
     || !check_type_optional(callback_args, &check_args, "callback_args", "tuple"))
    {
      SOLVE_CLEANUP_NULL();
    }
  if(callback_args != Py_None && callback_args != NULL)
    unpack_args(callback_args, &unpacked_args, &n_args);
  if(callback_kwargs == Py_None) callback_kwargs = NULL;
  if(PyArray_NDIM(x0) != 1)
    { //If x0 is not 1-dimensional then solve will fail and cause a segmentation fault.
      logger("[ERROR] x0 must be a 1-dimensional array");
      Py_XDECREF(x);
      Py_XDECREF(mL);
      Py_XDECREF(mU);
      Py_XDECREF(lambda);
      PyErr_SetString(PyExc_TypeError, "x0 passed to solve is not 1-dimensional.");
      SAFE_FREE(unpacked_args);
      return NULL;
    }
  
  bigfield->callback_args = unpacked_args;
  bigfield->n_callback_args = n_args;
  bigfield->callback_kwargs = callback_kwargs;
  if(nlp == NULL)
    {SOLVE_CLEANUP_TYPE("nlp objective passed to solve is NULL\n Problem created?\n")}
  if(bigfield->eval_h_python == NULL)
    AddIpoptStrOption(nlp, "hessian_approximation", "limited-memory");
  
  // allocate space for the initial point and set the values
  npy_intp *dim = PyArray_DIMS(x0);
  n = dim[0];
  dX[0] = n;
  
  x = (PyArrayObject*) PyArray_SimpleNew(1, dX, NPY_DOUBLE);
  if(!x) {SOLVE_CLEANUP_MEMORY()}
  newx0 = (Number*) malloc(sizeof(Number) * n);
  if(!newx0) {SOLVE_CLEANUP_MEMORY()}
  double *xdata = PyArray_DATA(x0);
  for(i=0; i<n; i++)
    newx0[i] = xdata[i];
  
  // Allocate multiplier arrays
  if(mL == NULL)
    mL_data = malloc(n * sizeof(double));
  else if(PyArray_DIMS(mL)[0] != n)
    {SOLVE_CLEANUP_TYPE("mult_x_L must be the same length as x0.\n");}
  else mL_data = PyArray_DATA(mL);
  if(mU == NULL)
    mU_data = malloc(n * sizeof(double));
  else if(PyArray_DIMS(mU)[0] != n)
    {SOLVE_CLEANUP_TYPE("mult_x_U must be the same length as x0.\n");}
  else mU_data = PyArray_DATA(mU);
  if(lambda == NULL)
    lambda_data = malloc(m * sizeof(double));
  else if(PyArray_DIMS(lambda)[0] != m)
    {SOLVE_CLEANUP_TYPE("mult_g must be the same length as the constraints.\n");}
  else lambda_data = PyArray_DATA(lambda);
  
  // For status code, see IpReturnCodes_inc.h in Ipopt
  status = IpoptSolve(nlp, newx0, NULL, &obj,
		      lambda_data, mL_data, mU_data,
		      (UserDataPtr)bigfield);
  if(lambda == NULL && lambda_data != NULL) free(lambda_data);
  if(mU == NULL && mU_data != NULL) free(mU_data);
  if(mL == NULL && mL_data != NULL) free(mL_data);
  double *return_x_data = PyArray_DATA(x);
  for(i=0; i<n; i++)
    return_x_data[i] = newx0[i];
  
  retval = Py_BuildValue("Odi", PyArray_Return(x), obj, status);
  // clean up and return

  Py_XDECREF(x);
  SAFE_FREE(newx0);
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
  if (kwargs == NULL || kwargs == Py_None || PyDict_Check(kwargs)) return TRUE;
  PyErr_Format(PyExc_RuntimeError, "C-API-Level Error: keywords are not of type dict");
  return FALSE;
}
Bool check_type_optional(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name)
{
  if(obj == NULL || checker(obj))
    return TRUE;
  PyErr_Format(PyExc_TypeError, "Wrong type for %s. Required: %s", obj_name, type_name);
  return FALSE;
}
Bool check_type(const PyObject *obj, Bool (*checker)(const PyObject*), const char *obj_name, const char *type_name)
{
  if(obj != NULL && check_type_optional(obj, checker, obj_name, type_name))
    return TRUE;
  PyErr_Format(PyExc_TypeError, "Error while parsing %s.", obj_name);
  return FALSE;
}
PyObject *close_model(PyObject *self, PyObject *args)
{
  problem *obj = (problem*) self;
  DispatchData *dp = obj->data;
  
  //Ungrab the callback functions because we do not need them anymore.
  Py_XDECREF(dp->eval_f_python);
  Py_XDECREF(dp->eval_grad_f_python);
  Py_XDECREF(dp->eval_g_python);
  Py_XDECREF(dp->eval_jac_g_python);
  Py_XDECREF(dp->eval_h_python);
  Py_XDECREF(dp->apply_new_python);
  sparsity_indices_free(&dp->sparsity_indices_jac_g);
  sparsity_indices_free(&dp->sparsity_indices_hess);
  
  free(dp);
  obj->data = NULL;
  
  FreeIpoptProblem(obj->nlp);
  obj->nlp = NULL;
  Py_INCREF(Py_True);
  return Py_True;
}

// Begin Python Module code section
static PyMethodDef ipoptMethods[] =
  {
   {"create", (PyCFunction)create, METH_VARARGS | METH_KEYWORDS, PYIPOPT_CREATE_DOC},
   {"set_loglevel", set_loglevel, METH_VARARGS, PYIPOPT_LOG_DOC},
   {NULL, NULL}
  };

#if PY_MAJOR_VERSION >= 3
#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF(ob, name, doc, methods)					\
  static struct PyModuleDef moduledef =					\
    { PyModuleDef_HEAD_INIT, name, doc, -1, methods, };			\
  ob = PyModule_Create(&moduledef);
#else
#define MOD_ERROR_VAL
#define MOD_SUCCESS_VAL(val)
#define MOD_INIT(name) void init##name(void)
#define MOD_DEF(ob, name, doc, methods)			\
  ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(pyipoptcore)
{
  PyObject * m;
  // Finish initialization of the problem type
  if(PyType_Ready(&IpoptProblemType) < 0) 
    return MOD_ERROR_VAL;
  
  MOD_DEF(m, "pyipoptcore", "A hook between Ipopt and Python", ipoptMethods);
    
  if(m == NULL)
    return MOD_ERROR_VAL;
  
  // Initialize numpy.
  // A segfault will occur if I use numarray without this..
  import_array();
  if(PyErr_Occurred())
    Py_FatalError("Unable to initialize module pyipoptcore");
  
  return MOD_SUCCESS_VAL(m);
}
