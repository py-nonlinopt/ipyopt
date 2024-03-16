#!/bin/env python3

"""Example for optimizing scipy.optimize.rosen."""

import numpy as np
import scipy.optimize

import ipyopt


def eval_f(x):
    """Directly evaluate the objective function f."""
    return scipy.optimize.rosen(x)


def eval_grad_f(x, out):
    """Evaluate the gradient of the objective function f."""
    out[()] = scipy.optimize.rosen_der(x)
    return out


def eval_g(_x, _out):
    """Evaluate the constraint functions.

    Constraints are defined by:
    g_L <= g(x) <= g_U
    """
    return


def eval_jac_g(_x, _out):
    """Evaluate the sparse Jacobian of constraint functions g."""
    return


# define the nonzero slots in the jacobian
# there are no nonzeros in the constraint jacobian
eval_jac_g_sparsity_indices = (np.array([]), np.array([]))


def eval_h(x, _lagrange, obj_factor, out):
    """Evaluate the sparse hessian of the Lagrangian.

    L = obj_factor * f + <lagrange, g>,
    where <.,.> denotes the inner product.
    """
    hess = scipy.optimize.rosen_hess(x)
    out[()] = hess[eval_h_sparsity_indices] * obj_factor
    return out


# there are maximum nonzeros (nvar*(nvar+1))/2 in the lagrangian hessian
eval_h_sparsity_indices = (
    np.array([0, 1, 1], dtype=int),
    np.array([0, 0, 1], dtype=int),
)


def main():
    """Entry point."""
    # define the parameters and their box constraints
    nvar = 2
    x_l = np.array([-3, -3], dtype=float)
    x_u = np.array([3, 3], dtype=float)

    # define the inequality constraints
    ncon = 0
    g_l = np.array([], dtype=float)
    g_u = np.array([], dtype=float)

    # create the nonlinear programming model
    nlp = ipyopt.Problem(
        nvar,
        x_l,
        x_u,
        ncon,
        g_l,
        g_u,
        eval_jac_g_sparsity_indices,
        eval_h_sparsity_indices,
        eval_f,
        eval_grad_f,
        eval_g,
        eval_jac_g,
        eval_h,
    )

    # define the initial guess
    x0 = np.array([-1.2, 1], dtype=float)

    # compute the results using ipopt
    results = nlp.solve(x0)

    # report the results
    print(results)


if __name__ == "__main__":
    main()
