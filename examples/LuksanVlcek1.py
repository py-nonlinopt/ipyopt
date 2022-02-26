#!/usr/bin/python

"""The same model as Ipopt/examples/ScalableProblems/LuksanVlcek1.cpp

You can set Ipopt options by calling ipyopt.Problem.set().
For instance, to set the tolarance, use

    nlp = ipyopt.Problem(...)
    nlp.set(tol=1e-8)

For a complete list of Ipopt options, use
    print(ipyopt.get_ipopt_options())
"""

import numpy
import ipyopt

nvar = 20
x_L = numpy.full((nvar,), -1e20)
x_U = numpy.full((nvar,), 1e20)

ncon = nvar - 2
_g_l = 1.0
_g_u = 1.0
g_L = numpy.full((ncon,), _g_l)
g_U = numpy.full((ncon,), _g_u)

x0 = numpy.empty((nvar,))
x0[0::2] = -1.2
x0[1::2] = 1.0


def eval_f(x):
    """Return the objective value"""
    assert len(x) == nvar
    return 100.0 * numpy.sum((x[:-1] ** 2 - x[1:]) ** 2) + numpy.sum(
        (x[:-1] - 1.0) ** 2
    )


def eval_grad_f(x, out):
    """Return the gradient of the objective"""
    assert len(x) == nvar
    out[0] = 0.0
    h = x[:-1] ** 2 - x[1:]
    out[1:] = -200.0 * h
    out[:-1] += 400.0 * x[:-1] * h + 2.0 * (x[:-1] - 1.0)
    return out


def eval_g(x, out):
    """Return the constraint residuals
    Constraints are defined by:
    g_L <= g(x) <= g_U
    """
    assert len(x) == nvar
    out[()] = (
        3.0 * x[1:-1] ** 3
        + 2.0 * x[2:]
        - 5.0
        + numpy.sin(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        + 4.0 * x[1:-1]
        - x[:-2] * numpy.exp(x[:-2] - x[1:-1])
        - 3.0
    )
    return out


def eval_jac_g(x, out):
    """Values of the jacobian of g"""
    assert len(x) == nvar
    out[::3] = -(1.0 + x[:-2]) * numpy.exp(x[:-2] - x[1:-1])
    out[1::3] = (
        9.0 * x[1:-1] ** 2
        + numpy.cos(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        + numpy.sin(x[1:-1] - x[2:]) * numpy.cos(x[1:-1] + x[2:])
        + 4.0
        + x[:-2] * numpy.exp(x[:-2] - x[1:-1])
    )
    out[2::3] = (
        2.0
        - numpy.cos(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        + numpy.sin(x[1:-1] - x[2:]) * numpy.cos(x[1:-1] + x[2:])
    )
    return out


# / * * * 0   ...       0 \
# | 0 * * * 0 ...       0 |
# | 0 0 * * * 0 ...     0 |
# |           ...         |
# \ 0 ...           * * * /
eval_jac_g_sparsity_indices = (
    numpy.repeat(numpy.arange(nvar - 2), 3),
    numpy.array(
        [numpy.arange(nvar - 2), numpy.arange(1, nvar - 1), numpy.arange(2, nvar)]
    ).T.flatten(),
)
# [0, 0, 0, 1, 1, 1, ...]
# [0, 1, 2, 1, 2, 3, ...]


def eval_h(x, lagrange, obj_factor, out):
    """Hessian of the Lagrangian
    L = obj_factor * f + <lagrange, g>,
    where <.,.> denotes the inner product.
    """
    out[-1] = 0.0
    out[:-2:2] = obj_factor * (2.0 + 400.0 * (3.0 * x[:-1] * x[:-1] - x[1:]))
    out[:-4:2] -= lagrange * (2.0 + x[:-2]) * numpy.exp(x[:-2] - x[1:-1])
    out[2::2] += obj_factor * 200.0
    out[2:-2:2] += lagrange * (
        18.0 * x[1:-1]
        - 2.0 * numpy.sin(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        + 2.0 * numpy.cos(x[1:-1] - x[2:]) * numpy.cos(x[1:-1] + x[2:])
        - x[:-2] * numpy.exp(x[:-2] - x[1:-1])
    )
    out[4::2] += lagrange * (
        -2.0 * numpy.sin(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        - 2.0 * numpy.cos(x[1:-1] - x[2:]) * numpy.cos(x[1:-1] + x[2:])
    )
    out[1::2] = obj_factor * (-400.0 * x[:-1])
    out[1:-2:2] += lagrange * (1.0 + x[:-2]) * numpy.exp(x[:-2] - x[1:-1])
    return out


# / * * 0 ...   0 \
# | 0 * * 0 ... 0 |
# |      ...      |
# | 0 ...   0 * * |
# \ 0 ...     0 * /
eval_h_sparsity_indices = (
    numpy.repeat(numpy.arange(nvar), 2)[: 2 * nvar - 1],
    numpy.array([numpy.arange(nvar), numpy.arange(1, nvar + 1)]).T.flatten()[
        : 2 * nvar - 1
    ],
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
    eval_h,
)

print(f"Going to call solve with x0 = {x0}")
zl = numpy.zeros(nvar)
zu = numpy.zeros(nvar)
constraint_multipliers = numpy.zeros(ncon)
_x, obj, status = nlp.solve(x0, mult_g=constraint_multipliers, mult_x_L=zl, mult_x_U=zu)


print("Solution of the primal variables, x")
print("x =", _x)

print("Solution of the bound multipliers, z_L and z_U")
print("z_L =", zl)
print("z_U =", zu)

print("Solution of the constraint multipliers, lambda")
print("lambda =", constraint_multipliers)

print("Objective value")
print(f"f(x*) = {obj}")
