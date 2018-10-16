/*
 * Copyright (c) 2008, Eric You Xu, Washington University All rights
 * reserved. Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following conditions
 * are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. * Redistributions in
 * binary form must reproduce the above copyright notice, this list of
 * conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution. * Neither the name of the
 * Washington University nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific
 * prior written permission.
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

/* 
 * Added "eval_intermediate_callback" by 
 * OpenMDAO at NASA Glenn Research Center, 2010 and 2011
 *
 * Changed logger from code contributed by alanfalloon  
*/

#include "hook.h"
#include <unistd.h>

void logger(const char *fmt, ...)
{
  if(user_log_level == VERBOSE)
    {
      va_list ap;
      va_start(ap, fmt);
      PySys_WriteStdout(fmt, ap);
      va_end(ap);
      PySys_WriteStdout("\n");
    }
}

PyObject *build_arg_tuple(PyObject ***arg_lists, unsigned int *n_args)
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
  DispatchData *myowndata = (DispatchData*) data;
  PyObject** callback_args = myowndata->callback_args;
  PyObject* callback_kwargs = myowndata->callback_kwargs;
  
  Bool result_as_bool;
  
  PyObject *arglist = NULL;
  PyObject *args[] = {
		      Py_BuildValue("i", alg_mod),
		      Py_BuildValue("i", iter_count),
		      Py_BuildValue("d", obj_value),
		      Py_BuildValue("d", inf_pr),
		      Py_BuildValue("d", inf_du),
		      Py_BuildValue("d", mu),
		      Py_BuildValue("d", d_norm),
		      Py_BuildValue("d", regularization_size),
		      Py_BuildValue("d", alpha_du),
		      Py_BuildValue("d", alpha_pr),
		      Py_BuildValue("i", ls_trials)
  };
  arglist = build_arg_tuple((PyObject**[]){args, callback_args, NULL},
			    (unsigned int[]){sizeof(args)/sizeof(PyObject*), myowndata->n_callback_args});
  
  PyObject *result = PyObject_Call(myowndata->eval_intermediate_callback_python, arglist, callback_kwargs);
  
  if(!result)
    PyErr_Print();
  
  result_as_bool = (Bool) PyLong_AsLong(result);
  
  Py_DECREF(result);
  Py_CLEAR(arglist);
  return result_as_bool;
}

Bool eval_f(Index n, Number *x, Bool new_x, Number *obj_value, UserDataPtr data)
{
  npy_intp dims[1];
  dims[0] = n;
  
  DispatchData *myowndata = (DispatchData*) data;
  PyObject **callback_args = myowndata->callback_args;
  PyObject *callback_kwargs = myowndata->callback_kwargs;
  
  import_array1(FALSE);
  PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*)x);
  if(!arrayx) return FALSE;
  
  if(new_x && myowndata->apply_new_python)
    {
      // Call the python function to applynew
      PyObject *arg1;
      arg1 = Py_BuildValue("(O)", arrayx);
      PyObject *tempresult = PyObject_CallObject(myowndata->apply_new_python, arg1);
      if(tempresult == NULL)
	{
	  logger("[Error] Python function apply_new returns NULL");
	  PyErr_Print();
	  Py_DECREF(arg1);
	  return FALSE;
	}
      Py_DECREF(arg1);
      Py_DECREF(tempresult);
    }
  PyObject* args[] = {arrayx};
  PyObject *arglist = build_arg_tuple((PyObject**[]){args, callback_args, NULL},
				      (unsigned int[]){sizeof(args)/sizeof(PyObject*), myowndata->n_callback_args});
  
  PyObject *result = PyObject_Call(myowndata->eval_f_python, arglist, callback_kwargs);
  
  if(result == NULL)
    {
      logger("[Error] Python function eval_f returns NULL");
      PyErr_Print();
      Py_DECREF(arrayx);
      Py_CLEAR(arglist);
      return FALSE;
    }
  
  *obj_value = PyFloat_AsDouble(result);
  
  if(PyErr_Occurred())
    {
      logger("[Error] Python function eval_f returns non-PyFloat");
      PyErr_Print();
      Py_DECREF(result);
      Py_DECREF(arrayx);
      Py_CLEAR(arglist);
      return FALSE;
    }
  
  Py_DECREF(result);
  Py_DECREF(arrayx);
  Py_CLEAR(arglist);
  return TRUE;
}

Bool eval_grad_f(Index n, Number *x, Bool new_x, Number *grad_f, UserDataPtr data)
{
  DispatchData *myowndata = (DispatchData*) data;
  PyObject** callback_args = myowndata->callback_args;
  PyObject* callback_kwargs = myowndata->callback_kwargs;
  
  if(myowndata->eval_grad_f_python == NULL)
    PyErr_Print();

  npy_intp dims[1];
  dims[0] = n;

  import_array1(FALSE);

  PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*)x);
  if(!arrayx) return FALSE;

  if(new_x && myowndata->apply_new_python)
    {
      // Call the python function to applynew
      PyObject *arg1 = Py_BuildValue("(O)", arrayx);
      PyObject *tempresult = PyObject_CallObject(myowndata->apply_new_python, arg1);
      if(tempresult == NULL)
	{
	  logger("[Error] Python function apply_new returns NULL");
	  PyErr_Print();
	  Py_DECREF(arg1);
	  return FALSE;
	}
      Py_DECREF(arg1);
      Py_DECREF(tempresult);
    }

  PyObject *args[] = {arrayx};
  PyObject *arglist = build_arg_tuple((PyObject**[]){args, callback_args, NULL},
				      (unsigned int[]){sizeof(args)/sizeof(PyObject*), myowndata->n_callback_args});
  
  PyArrayObject *result = (PyArrayObject*) PyObject_Call(myowndata->eval_grad_f_python, arglist, callback_kwargs);
  
  if(result == NULL)
    {
      logger("[Error] Python function eval_grad_f returns NULL");
      PyErr_Print();
      return FALSE;
    }
  
  if(!PyArray_Check(result))
    {
      logger("[Error] Python function eval_grad_f returns non-PyArray");
      Py_DECREF(result);
      return FALSE;
    }

  double *tempdata = (double*)result->data;
  int i;
  for(i=0; i<n; i++) grad_f[i] = tempdata[i];
  
  Py_DECREF(result);
  Py_CLEAR(arrayx);
  Py_CLEAR(arglist);
  return TRUE;
}

Bool eval_g(Index n, Number *x, Bool new_x, Index m, Number *g, UserDataPtr data)
{
  DispatchData *myowndata = (DispatchData*) data;
  PyObject** callback_args = myowndata->callback_args;
  PyObject* callback_kwargs = myowndata->callback_kwargs;
  
  if(myowndata->eval_g_python == NULL) PyErr_Print();
  npy_intp dims[1];
  int i;
  double *tempdata;

  dims[0] = n;
  import_array1(FALSE);
  
  PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*)x);
  if(!arrayx) return FALSE;
  
  if(new_x && myowndata->apply_new_python)
    {
      // Call the python function to applynew
      PyObject *arg1 = Py_BuildValue("(O)", arrayx);
      PyObject *tempresult = PyObject_CallObject(
						 myowndata->apply_new_python, arg1);
      if(tempresult == NULL)
	{
	  logger("[Error] Python function apply_new returns NULL");
	  PyErr_Print();
	  Py_DECREF(arg1);
	  return FALSE;
	}
      Py_DECREF(arg1);
      Py_DECREF(tempresult);
    }

  PyObject *args[] = {arrayx};
  PyObject *arglist = build_arg_tuple((PyObject**[]){args, callback_args, NULL},
				      (unsigned int[]){sizeof(args)/sizeof(PyObject*), myowndata->n_callback_args});
  
  PyArrayObject *result = (PyArrayObject*) PyObject_Call(myowndata->eval_g_python,
							  arglist, callback_kwargs);

  if(result == NULL)
    {
      logger("[Error] Python function eval_g returns NULL");
      PyErr_Print();
      return FALSE;
    }
  
  if(!PyArray_Check(result))
    {
      logger("[Error] Python function eval_g returns non-PyArray");
      Py_DECREF(result);
      return FALSE;
    }

  tempdata = (double*)result->data;
  for(i=0; i<m; i++) g[i] = tempdata[i];

  Py_DECREF(result);
  Py_CLEAR(arrayx);
  Py_CLEAR(arglist);
  return TRUE;
}

Bool eval_jac_g(Index n, Number *x, Bool new_x,
		Index m, Index nele_jac,
		Index *iRow, Index *jCol, Number *values, UserDataPtr data)
{
  DispatchData *myowndata = (DispatchData*) data;
  PyObject** callback_args = myowndata->callback_args;
  PyObject* callback_kwargs = myowndata->callback_kwargs;
  
  unsigned int i;

  npy_intp dims[1];
  dims[0] = n;

  double *tempdata;

  if(myowndata->eval_grad_f_python == NULL) PyErr_Print();

  if(values == NULL)
    for(i=0; i<myowndata->sparsity_indices_jac_g.n; i++)
      {
	iRow[i] = myowndata->sparsity_indices_jac_g.row[i];
	jCol[i] = myowndata->sparsity_indices_jac_g.col[i];
      }
  else
    {
      PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*)x);
      if(!arrayx) return FALSE;
      if(new_x && myowndata->apply_new_python)
	{
	  // Call the python function to applynew
	  PyObject *arg1 = Py_BuildValue("(O)", arrayx);
	  PyObject *tempresult = PyObject_CallObject(myowndata->apply_new_python, arg1);
	  if(tempresult == NULL)
	    {
	      logger("[Error] Python function apply_new returns NULL");
	      Py_DECREF(arg1);
	      return FALSE;
	    }
	  Py_DECREF(arg1);
	  Py_DECREF(tempresult);
	}
      PyObject *args[] = {arrayx};
      PyObject *arglist = build_arg_tuple((PyObject**[]){args, callback_args, NULL},
					  (unsigned int[]){sizeof(args)/sizeof(PyObject*), myowndata->n_callback_args});

      PyArrayObject *result = (PyArrayObject*) PyObject_Call(myowndata->eval_jac_g_python, arglist, callback_kwargs);

      if(result == NULL)
	{
	  logger("[Error] Python function eval_jac_g returns NULL");
	  PyErr_Print();
	  return FALSE;
	}

      if(!PyArray_Check(result))
	{
	  logger("[Error] Python function eval_jac_g returns non-PyArray");
	  Py_DECREF(result);
	  return FALSE;
	}

      /*
       * Code is buggy here. We assume that result is a double
       * array
       */
      assert(result->descr->type == 'd');
      tempdata = (double*)result->data;
      
      for(i=0; i<(unsigned int)nele_jac; i++)
	values[i] = tempdata[i];
      
      Py_DECREF(result);
      Py_CLEAR(arrayx);
      Py_CLEAR(arglist);
    }
  return TRUE;
}

Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
	    Index m, Number *lambda, Bool new_lambda,
	    Index nele_hess, Index *iRow, Index *jCol,
	    Number *values, UserDataPtr data)
{
  DispatchData *myowndata = (DispatchData*) data;
  PyObject** callback_args = myowndata->callback_args;
  PyObject* callback_kwargs = myowndata->callback_kwargs;
  
  unsigned int i;
  npy_intp dims[1];
  npy_intp dims2[1];
  
  if(myowndata->eval_h_python == NULL)
    {
      logger("[Error] There is no eval_h assigned");
      return FALSE;
    }
  PyObject *arglist;
  if(values == NULL)
    for(i=0; i<myowndata->sparsity_indices_hess.n; i++)
      {
	iRow[i] = myowndata->sparsity_indices_hess.row[i];
	jCol[i] = myowndata->sparsity_indices_hess.col[i];
      }
  else
    {
      PyObject *objfactor = Py_BuildValue("d", obj_factor);
      
      dims[0] = n;
      PyObject *arrayx = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*)x);
      if(!arrayx) return FALSE;

      if(new_x && myowndata->apply_new_python)
	{
	  // Call the python function to applynew
	  PyObject *arg1 = Py_BuildValue("(O)", arrayx);
	  PyObject *tempresult = PyObject_CallObject(myowndata->apply_new_python, arg1);
	  if(tempresult == NULL)
	    {
	      logger("[Error] Python function apply_new returns NULL");
	      PyErr_Print();
	      Py_DECREF(arg1);
	      return FALSE;
	    }
	  Py_DECREF(arg1);
	  Py_DECREF(tempresult);
	}
      dims2[0] = m;
      PyObject *lagrangex = PyArray_SimpleNewFromData(1, dims2, PyArray_DOUBLE, (char*)lambda);
      if(!lagrangex) return FALSE;
      
      PyObject *c_arg_list[] = {arrayx, lagrangex, objfactor};

      arglist = build_arg_tuple((PyObject**[]){c_arg_list, callback_args, NULL},
				(unsigned int[]){sizeof(c_arg_list)/sizeof(PyObject*), myowndata->n_callback_args});
      PyArrayObject *result = (PyArrayObject*) PyObject_Call(myowndata->eval_h_python, arglist, callback_kwargs);
      
      if(result == NULL)
	{
	  logger("[Error] Python function eval_h returns NULL");
	  PyErr_Print();
	  return FALSE;
	}
      
      if(!PyArray_Check(result))
	{
	  logger("[Error] Python function eval_h returns non-PyArray");
	  Py_DECREF(result);
	  return FALSE;
	}

      double *tempdata = (double*)result->data;
      for(i=0; i<(unsigned int)nele_hess; i++)
	values[i] = tempdata[i];
      
      Py_CLEAR(arrayx);
      Py_CLEAR(lagrangex);
      Py_CLEAR(objfactor);
      Py_DECREF(result);
      Py_CLEAR(arglist);
    }
  return TRUE;
}
