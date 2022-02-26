#!/bin/env python3

"""Example for optimizing scipy.optimize.rosen"""

from typing import Tuple
import numpy
import scipy.optimize
import ipyopt


def eval_f(X):
    """Directly evaluate the objective function f."""
    return scipy.optimize.rosen(X)


def eval_grad_f(X, out):
    """Evaluate the gradient of the objective function f."""
    out[()] = scipy.optimize.rosen_der(X)
    return out


def eval_g(_X, _out):
    """Evaluate the constraint functions.
    Constraints are defined by:
    g_L <= g(x) <= g_U
    """
    return


def eval_jac_g(_X, _out):
    """Evaluate the sparse Jacobian of constraint functions g.

    @param X: parameter values
    @param out: The numpy array to write the result values into
    """
    return


# define the nonzero slots in the jacobian
# there are no nonzeros in the constraint jacobian
eval_jac_g_sparsity_indices: Tuple[numpy.ndarray, numpy.ndarray] = (
    numpy.array([]),
    numpy.array([]),
)


def eval_h(X, _lagrange, obj_factor, out):
    """Evaluate the sparse hessian of the Lagrangian
    L = obj_factor * f + <lagrange, g>,
    where <.,.> denotes the inner product.

    @param X: parameter values
    @param lagrange: something about the constraints
    @param obj_factor: no clue what this is
    @param out: The numpy array to write the result values into
    """
    H = scipy.optimize.rosen_hess(X)
    out[()] = H[eval_h_sparsity_indices] * obj_factor
    return out


# there are maximum nonzeros (nvar*(nvar+1))/2 in the lagrangian hessian
eval_h_sparsity_indices = (
    numpy.array([0, 1, 1], dtype=int),
    numpy.array([0, 0, 1], dtype=int),
)


def main():  # pylint: disable=missing-function-docstring
    # define the parameters and their box constraints
    nvar = 2
    x_L = numpy.array([-3, -3], dtype=float)
    x_U = numpy.array([3, 3], dtype=float)

    # define the inequality constraints
    ncon = 0
    g_L = numpy.array([], dtype=float)
    g_U = numpy.array([], dtype=float)

    # create the nonlinear programming model
    nlp = ipyopt.Problem(
        nvar,
        x_L,
        x_U,
        ncon,
        g_L,
        g_U,
        eval_jac_g_sparsity_indices,
        eval_h_sparsity_indices,
        eval_f,
        eval_grad_f,
        eval_g,
        eval_jac_g,
        eval_h,
    )

    # define the initial guess
    x0 = numpy.array([-1.2, 1], dtype=float)

    # compute the results using ipopt
    results = nlp.solve(x0)

    # report the results
    print(results)


if __name__ == "__main__":
    main()
