#!/usr/bin/python

"""Same as hs071.py, but this time with warm start"""

from numpy import ones, float_, array, zeros
import ipyopt


nvar = 4
x_L = ones((nvar), dtype=float_) * 1.0
x_U = ones((nvar), dtype=float_) * 5.0

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


eval_jac_g_sparsity_indices = (
    array([0, 0, 0, 0, 1, 1, 1, 1]),
    array([0, 1, 2, 3, 0, 1, 2, 3]),
)


def eval_h(x, lagrange, obj_factor):
    """Hessian of the Lagrangian
    L = obj_factor * f + <lagrange, g>,
    where <.,.> denotes the inner product.
    """
    values = zeros((10), float_)
    values[0] = obj_factor * (2 * x[3])
    values[1] = obj_factor * (x[3])
    values[2] = 0
    values[3] = obj_factor * (x[3])
    values[4] = 0
    values[5] = 0
    values[6] = obj_factor * (2 * x[0] + x[1] + x[2])
    values[7] = obj_factor * (x[0])
    values[8] = obj_factor * (x[0])
    values[9] = 0
    values[1] += lagrange[0] * (x[2] * x[3])

    values[3] += lagrange[0] * (x[1] * x[3])
    values[4] += lagrange[0] * (x[0] * x[3])

    values[6] += lagrange[0] * (x[1] * x[2])
    values[7] += lagrange[0] * (x[0] * x[2])
    values[8] += lagrange[0] * (x[0] * x[1])
    values[0] += lagrange[1] * 2
    values[2] += lagrange[1] * 2
    values[5] += lagrange[1] * 2
    values[9] += lagrange[1] * 2
    return values


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
pi0 = array([1.0, 1.0])

print("Going to call solve for 4 iterations")
print(f"x0 = {x0}")
nlp.set(max_iter=4)  # limit the number of max iterations
zl = zeros(nvar)
zu = zeros(nvar)
constraint_multipliers = zeros(ncon)
_x, obj, status = nlp.solve(x0, mult_g=constraint_multipliers, mult_x_L=zl, mult_x_U=zu)

print("Solution of the bound multipliers, z_L and z_U")
print("z_L =", zl)
print("z_U =", zu)

print("Solution of the constraint multipliers, lambda")
print("lambda =", constraint_multipliers)


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
nlp.set(
    warm_start_init_point="yes",
    warm_start_bound_push=1e-8,
    warm_start_slack_bound_push=1e-8,
    warm_start_mult_bound_push=1e-8,
    print_level=5,
)
print("Starting at previous solution and solving again")
_x, obj, status = nlp.solve(_x, mult_g=constraint_multipliers, mult_x_L=zl, mult_x_U=zu)

print("Solution of the primal variables, x")
print("x =", _x)

print("Solution of the bound multipliers, z_L and z_U")
print("z_L =", zl)
print("z_U =", zu)

print("Solution of the constraint multipliers, lambda")
print("lambda =", constraint_multipliers)

print("Objective value")
print(f"f(x*) = {obj}")
