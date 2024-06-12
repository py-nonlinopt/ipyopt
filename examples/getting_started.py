#!/bin/env python3

"""Example for the getting started section in the docs."""

import numpy as np
from numpy.typing import NDArray

import ipyopt


def f(x: NDArray[np.float64]) -> float:
    """Objective value."""
    out: float = np.sum(x**2)
    return out


def grad_f(x: NDArray[np.float64], out: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gradient of the objective."""
    out[()] = 2.0 * x
    return out


def g(x: NDArray[np.float64], out: NDArray[np.float64]) -> NDArray[np.float64]:
    """Constraint function: squared distance to (1, 0, ..., 0)."""
    out[0] = (x[0] - 1.0) ** 2 + np.sum(x[1:] ** 2)
    return out


def jac_g(x: NDArray[np.float64], out: NDArray[np.float64]) -> NDArray[np.float64]:
    """Jacobian of the constraint function."""
    out[()] = 2.0 * x
    out[0] -= 2.0
    return out


def hess_g(
    x: NDArray[np.float64],
    lagrange: NDArray[np.float64],
    obj_factor: float,
    out: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Hessian of the constraint function."""
    out[()] = np.full_like(x, 2.0 * (obj_factor + lagrange[0]))
    return out


nlp = ipyopt.Problem(
    n=3,
    x_l=np.array([-10.0, -10.0, -10.0]),
    x_u=np.array([10.0, 10.0, 10.0]),
    m=1,
    g_l=np.array([0.0]),
    g_u=np.array([4.0]),
    sparsity_indices_jac_g=(np.array([0, 0, 0]), np.array([0, 1, 2])),
    sparsity_indices_h=(np.array([0, 1, 2]), np.array([0, 1, 2])),
    eval_f=f,
    eval_grad_f=grad_f,
    eval_g=g,
    eval_jac_g=jac_g,
    eval_h=hess_g,
)

x, obj, status = nlp.solve(x0=np.array([0.1, 0.1, 0.1]))


print("Solution of the primal variables, x")
print(f"x = {x}")

print("Objective value")
print(f"f(x*) = {obj}")
