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
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL ipyopt_ARRAY_API
#include "numpy/arrayobject.h"
#include "callback.h"

static PyObject *build_arg_tuple(PyObject ***arg_lists, unsigned int *n_args)
/*
 *  This is a convenience replacement for Py_BuildValue.
 *  It takes a NULL terminated array arg_lists of c arrays of PyObject* as arguments
 *  having a number of arguments accorging to the n_args array.
 *  It constructs a new tuple consisting of all elements in all arrays in arg_lists.
 */
{
  unsigned int i, j, l;
  unsigned int size_tuple = 0;
  for(i=0; arg_lists[i] != NULL; i++)
    size_tuple += n_args[i];
  PyObject *tuple = PyTuple_New(size_tuple);
  l = 0;
  for(i=0; arg_lists[i] != NULL; i++)
    for(j=0; j<n_args[i]; j++)
      {
	Py_INCREF(arg_lists[i][j]);
	PyTuple_SET_ITEM(tuple, l++, arg_lists[i][j]);
      }
  return tuple;
}

Bool eval_intermediate_callback(Index alg_mod,	// 0 is regular, 1 is resto
				Index iter_count, Number obj_value,
				Number inf_pr, Number inf_du,
				Number mu, Number d_norm,
				Number regularization_size,
				Number alpha_du, Number alpha_pr,
				Index ls_trials, UserDataPtr data)
{
  DispatchData *callback_data = (DispatchData*) data;
  PyObject** callback_args = callback_data->callback_args;
  PyObject* callback_kwargs = callback_data->callback_kwargs;
  
  Bool result_as_bool;
  
  PyObject *args[] = {
		      PyLong_FromLong(alg_mod),
		      PyLong_FromLong(iter_count),
		      PyFloat_FromDouble(obj_value),
		      PyFloat_FromDouble(inf_pr),
		      PyFloat_FromDouble(inf_du),
		      PyFloat_FromDouble(mu),
		      PyFloat_FromDouble(d_norm),
		      PyFloat_FromDouble(regularization_size),
		      PyFloat_FromDouble(alpha_du),
		      PyFloat_FromDouble(alpha_pr),
		      PyLong_FromLong(ls_trials)
  };
  PyObject *arglist = build_arg_tuple((PyObject**[]){args, callback_args, NULL},
				      (unsigned int[]){sizeof(args)/sizeof(PyObject*), callback_data->n_callback_args});
  
  PyObject *result = PyObject_Call(callback_data->eval_intermediate_callback_python, arglist, callback_kwargs);
  
  if(!result)
    PyErr_Print();
  
  result_as_bool = (Bool)PyLong_AsLong(result);
  
  Py_DECREF(result);
  Py_CLEAR(arglist);
  return result_as_bool;
}

static Bool apply_new(PyObject *apply_new_python, PyObject *x_arr)
{
  if(apply_new_python == NULL) return TRUE;
  // Call the python function to applynew
  PyObject *arg1 = PyTuple_Pack(1, x_arr);
  PyObject *tempresult = PyObject_CallObject(apply_new_python, arg1);
  if(PyErr_Occurred() != NULL || tempresult == NULL)
    {
      if(PyErr_Occurred() == NULL)
	PyErr_Format(PyExc_RuntimeError, "Python function apply_new returns NULL");
      Py_DECREF(arg1);
      return FALSE;
    }
  Py_DECREF(arg1);
  Py_DECREF(tempresult);
  return TRUE;
}

static PyObject* call(PyObject *callback, PyObject* args[], unsigned int n_args,
		 PyObject **callback_args, unsigned int n_callback_args,
		 PyObject *callback_kwargs)
{
  if(callback == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "python callback is NULL");
      return NULL;
    }
  PyObject *arglist = build_arg_tuple((PyObject**[]){args, callback_args, NULL},
				      (unsigned int[]){n_args, n_callback_args});
  
  PyObject *result = PyObject_Call(callback, arglist, callback_kwargs);
  Py_CLEAR(arglist);
  return result;
}
static PyObject *wrap_array(Number *x, unsigned int n)
{
  npy_intp dims[] = {n};
  PyObject *x_arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (char*)x);
  return x_arr;
}
static Bool check_objects_2(PyObject *o1, PyObject *o2)
{
  if(o1 != NULL && o2 != NULL) return TRUE;
  if(o1 != NULL) Py_CLEAR(o1);
  if(o2 != NULL) Py_CLEAR(o2);
  return FALSE;
}
static Bool check_objects_3(PyObject *o1, PyObject *o2, PyObject *o3)
{
  if(o1 != NULL && o2 != NULL && o3 != NULL) return TRUE;
  if(o1 != NULL) Py_CLEAR(o1);
  if(o2 != NULL) Py_CLEAR(o2);
  if(o3 != NULL) Py_CLEAR(o3);
  return FALSE;
}

Bool eval_f(Index n, Number *x, Bool new_x, Number *obj_value, UserDataPtr data)
{
  DispatchData *callback_data = (DispatchData*)data;
  PyObject **callback_args = callback_data->callback_args;
  PyObject *callback_kwargs = callback_data->callback_kwargs;
  
  PyObject *x_arr = wrap_array(x, n);
  PyObject* args[] = {x_arr};
  PyObject *return_value = NULL;
  if(!x_arr) return FALSE;
  if(!new_x || apply_new(callback_data->apply_new_python, x_arr))
    return_value = call(callback_data->eval_f_python, args, sizeof(args)/sizeof(PyObject*), callback_args, callback_data->n_callback_args, callback_kwargs);
  Py_DECREF(x_arr);
  
  if(return_value == NULL)
    return FALSE;
  
  *obj_value = PyFloat_AsDouble(return_value);
  Py_DECREF(return_value);
  
  if(PyErr_Occurred())
    {
      PyErr_Format(PyExc_RuntimeError, "Python function eval_f returns non-PyFloat");
      return FALSE;
    }
  
  return TRUE;
}

Bool eval_grad_f(Index n, Number *x, Bool new_x, Number *grad_f, UserDataPtr data)
{
  DispatchData *callback_data = (DispatchData*) data;
  PyObject** callback_args = callback_data->callback_args;
  PyObject* callback_kwargs = callback_data->callback_kwargs;
  PyObject *return_value = NULL;
  
  PyObject *x_arr = wrap_array(x, n);
  PyObject *out = wrap_array(grad_f, n);
  if(!check_objects_2(x_arr, out)) return FALSE;
  PyObject *args[] = {x_arr, out};

  if(!new_x || apply_new(callback_data->apply_new_python, x_arr))
    return_value = call(callback_data->eval_grad_f_python, args, sizeof(args)/sizeof(PyObject*), callback_args, callback_data->n_callback_args, callback_kwargs);
  Py_CLEAR(x_arr);
  Py_CLEAR(out);

  return return_value != NULL;
}

Bool eval_g(Index n, Number *x, Bool new_x, Index m, Number *g, UserDataPtr data)
{
  DispatchData *callback_data = (DispatchData*) data;
  PyObject** callback_args = callback_data->callback_args;
  PyObject* callback_kwargs = callback_data->callback_kwargs;
  PyObject *return_value = NULL;
  
  PyObject *x_arr = wrap_array(x, n);
  PyObject *out = wrap_array(g, m);
  if(!check_objects_2(x_arr, out)) return FALSE;
  PyObject *args[] = {x_arr, out};
  
  if(!new_x || apply_new(callback_data->apply_new_python, x_arr))
    return_value = call(callback_data->eval_g_python, args, sizeof(args)/sizeof(PyObject*), callback_args, callback_data->n_callback_args, callback_kwargs);

  Py_CLEAR(x_arr);
  Py_CLEAR(out);

  return return_value != NULL;
}

static void set_sparsity(Index *iRow, Index *jCol, SparsityIndices *sparsity_indices)
{
  unsigned int i;
  for(i=0; i<sparsity_indices->n; i++)
    {
      iRow[i] = sparsity_indices->row[i];
      jCol[i] = sparsity_indices->col[i];
    }
}

Bool eval_jac_g(Index n, Number *x, Bool new_x,
		Index m, Index nele_jac,
		Index *iRow, Index *jCol, Number *values, UserDataPtr data)
{
  DispatchData *callback_data = (DispatchData*) data;
  PyObject** callback_args = callback_data->callback_args;
  PyObject* callback_kwargs = callback_data->callback_kwargs;
  PyObject *return_value = NULL;
  
  if(values == NULL)
    {
      set_sparsity(iRow, jCol, &callback_data->sparsity_indices_jac_g);
      return TRUE;
    }
  PyObject *x_arr = wrap_array(x, n);
  PyObject *out = wrap_array(values, nele_jac);
  if(!check_objects_2(x_arr, out)) return FALSE;
  PyObject *args[] = {x_arr, out};
  if(!new_x || apply_new(callback_data->apply_new_python, x_arr))
    return_value = call(callback_data->eval_jac_g_python, args, sizeof(args)/sizeof(PyObject*), callback_args, callback_data->n_callback_args, callback_kwargs);
  
  Py_CLEAR(x_arr);
  Py_CLEAR(out);

  return return_value != NULL;
}

Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
	    Index m, Number *lambda, Bool new_lambda,
	    Index nele_hess, Index *iRow, Index *jCol,
	    Number *values, UserDataPtr data)
{
  DispatchData *callback_data = (DispatchData*) data;
  PyObject** callback_args = callback_data->callback_args;
  PyObject* callback_kwargs = callback_data->callback_kwargs;
  PyObject *return_value = NULL;
  
  if(values == NULL)
    {
      set_sparsity(iRow, jCol, &callback_data->sparsity_indices_hess);
      return TRUE;
    }

  PyObject *objfactor = PyFloat_FromDouble(obj_factor);
  PyObject *x_arr = wrap_array(x, n);
  PyObject *out = wrap_array(values, nele_hess);
  PyObject *lagrangex = wrap_array(lambda, m);
  if(!check_objects_3(x_arr, out, lagrangex)) return FALSE;
  
  PyObject *args[] = {x_arr, lagrangex, objfactor, out};
  if(!new_x || apply_new(callback_data->apply_new_python, x_arr))
    return_value = call(callback_data->eval_h_python, args, sizeof(args)/sizeof(PyObject*), callback_args, callback_data->n_callback_args, callback_kwargs);
  
  Py_CLEAR(x_arr);
  Py_CLEAR(out);
  Py_CLEAR(lagrangex);
  Py_CLEAR(objfactor);
  return return_value != NULL;
}
