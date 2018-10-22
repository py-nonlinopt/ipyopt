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

#include "Python.h"
#include "IpStdCInterface.h"

#ifndef PY_IPOPT_HOOK_
#define PY_IPOPT_HOOK_

// A series of callback functions used by Ipopt C Interface
Bool eval_f(Index n,
	    Number * x, Bool new_x, Number * obj_value, UserDataPtr user_data);

Bool eval_grad_f(Index n,
		 Number * x,
		 Bool new_x, Number * grad_f, UserDataPtr user_data);

Bool eval_g(Index n,
	    Number * x, Bool new_x, Index m, Number * g, UserDataPtr user_data);

Bool eval_jac_g(Index n, Number * x, Bool new_x,
		Index m, Index nele_jac,
		Index * iRow, Index * jCol, Number * values,
		UserDataPtr user_data);

Bool eval_h(Index n, Number * x, Bool new_x, Number obj_factor,
	    Index m, Number * lambda, Bool new_lambda,
	    Index nele_hess, Index * iRow, Index * jCol,
	    Number * values, UserDataPtr user_data);

Bool eval_intermediate_callback(Index alg_mod,
				Index iter_count, Number obj_value,
				Number inf_pr, Number inf_du,
				Number mu, Number d_norm,
				Number regularization_size,
				Number alpha_du, Number alpha_pr,
				Index ls_trials, UserDataPtr data);

typedef struct
{
  unsigned int n;
  Index *row, *col;
} SparsityIndices;

typedef struct
{
  PyObject *eval_f_python;
  PyObject *eval_grad_f_python;
  PyObject *eval_g_python;
  PyObject *eval_jac_g_python;
  PyObject *eval_h_python;
  PyObject *apply_new_python;
  PyObject *eval_intermediate_callback_python;
  unsigned int n_callback_args;
  PyObject **callback_args;
  PyObject *callback_kwargs;
  SparsityIndices sparsity_indices_jac_g, sparsity_indices_hess;
} DispatchData;


#endif				//  PY_IPOPT_HOOK_
