#!/bin/env python3

"""The same model as Ipopt/examples/hs071

You can set Ipopt options by calling ipyopt.Problem.set().
For instance, to set the tolarance, use

    nlp = ipyopt.Problem(...)
    nlp.set(tol=1e-8)

For a complete list of Ipopt options, use
    print(ipyopt.get_ipopt_options())
"""

from numpy import ones, float_, array, zeros
import ipyopt

nvar = 4
x_L = ones(nvar, dtype=float_) * 1.0
x_U = ones(nvar, dtype=float_) * 5.0

ncon = 2

g_L = array([25.0, 40.0])
g_U = array([2.0e19, 40.0])


def eval_f(x):
    """Return the objective value"""
    assert len(x) == nvar
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


def eval_grad_f(x, out):
    """Return the gradient of the objective"""
    assert len(x) == nvar
    out[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2])
    out[1] = x[0] * x[3]
    out[2] = x[0] * x[3] + 1.0
    out[3] = x[0] * (x[0] + x[1] + x[2])
    return out


def eval_g(x, out):
    """Return the constraint residuals
    Constraints are defined by:
    g_L <= g(x) <= g_U
    """
    assert len(x) == nvar
    out[0] = x[0] * x[1] * x[2] * x[3]
    out[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
    return out


def eval_jac_g(x, out):
    """Values of the jacobian of g"""
    assert len(x) == nvar
    out[()] = [
        x[1] * x[2] * x[3],
        x[0] * x[2] * x[3],
        x[0] * x[1] * x[3],
        x[0] * x[1] * x[2],
        2.0 * x[0],
        2.0 * x[1],
        2.0 * x[2],
        2.0 * x[3],
    ]
    return out


# The following declares, that jac_g only has non zero
# entries at "*" (in this case no zeros):
# / * * * * \
# \ * * * * /
eval_jac_g_sparsity_indices = (
    array([0, 0, 0, 0, 1, 1, 1, 1]),
    array([0, 1, 2, 3, 0, 1, 2, 3]),
)


def eval_h(x, lagrange, obj_factor, out):
    """Hessian of the Lagrangian
    L = obj_factor * f + <lagrange, g>,
    where <.,.> denotes the inner product.
    """
    out[0] = obj_factor * (2 * x[3])
    out[1] = obj_factor * (x[3])
    out[2] = 0
    out[3] = obj_factor * (x[3])
    out[4] = 0
    out[5] = 0
    out[6] = obj_factor * (2 * x[0] + x[1] + x[2])
    out[7] = obj_factor * (x[0])
    out[8] = obj_factor * (x[0])
    out[9] = 0
    out[1] += lagrange[0] * (x[2] * x[3])

    out[3] += lagrange[0] * (x[1] * x[3])
    out[4] += lagrange[0] * (x[0] * x[3])

    out[6] += lagrange[0] * (x[1] * x[2])
    out[7] += lagrange[0] * (x[0] * x[2])
    out[8] += lagrange[0] * (x[0] * x[1])
    out[0] += lagrange[1] * 2
    out[2] += lagrange[1] * 2
    out[5] += lagrange[1] * 2
    out[9] += lagrange[1] * 2
    return out


# The following declares, that h only has non zero
# entries at "*":
# / * 0 0 0 \
# | * * 0 0 |
# | * * * 0 |
# \ * * * * /
eval_h_sparsity_indices = (
    array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
    array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]),
)


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
)

x0 = array([1.0, 5.0, 5.0, 1.0])

print(f"Going to call solve with x0 = {x0}")
zl = zeros(nvar)
zu = zeros(nvar)
constraint_multipliers = zeros(ncon)

_x, obj, status = nlp.solve(x0, mult_g=constraint_multipliers, mult_x_L=zl, mult_x_U=zu)
# NOTE: x0 is mutated so that x0 is now equal to the solution _x

print("Solution of the primal variables, x")
print("x =", _x)

print("Solution of the bound multipliers, z_L and z_U")
print("z_L =", zl)
print("z_U =", zu)

print("Solution of the constraint multipliers, lambda")
print("lambda =", constraint_multipliers)

print("Objective value")
print(f"f(x*) = {obj}")
