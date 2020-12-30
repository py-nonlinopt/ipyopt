#!/usr/bin/python

"""
The same model as Ipopt/examples/ScalableProblems/LuksanVlcek1.cpp

You can set Ipopt options by calling nlp.set.
For instance, to set the tolarance by calling

    nlp.set(tol=1e-8)

For a complete list of Ipopt options, refer to
    http://www.coin-or.org/Ipopt/documentation/node59.html
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
x0[1::2] = 1.


def eval_f(x):
    assert len(x) == nvar
    return (100. * numpy.sum((x[:-1]**2 - x[1:])**2) +
            numpy.sum((x[:-1] - 1.)**2))


def eval_grad_f(x, out):
    assert len(x) == nvar
    out[0] = 0.
    h = (x[:-1]**2 - x[1:])
    out[1:] = -200.* h
    out[:-1] += 400.* x[:-1]*h + 2.*(x[:-1] - 1.)
    return out


def eval_g(x, out):
    assert len(x) == nvar
    out[()] = (3. * x[1:-1]**3 + 2. * x[2:] - 5.
              + numpy.sin(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
              + 4. * x[1:-1]
              - x[:-2] * numpy.exp(x[:-2] - x[1:-1]) - 3.)
    return out


def eval_jac_g(x, out):
    assert len(x) == nvar
    out[::3] = -(1. + x[:-2]) * numpy.exp(x[:-2] - x[1:-1])
    out[1::3] = (
        9. * x[1:-1]**2
        + numpy.cos(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        + numpy.sin(x[1:-1] - x[2:]) * numpy.cos(x[1:-1] + x[2:])
        + 4. + x[:-2] * numpy.exp(x[:-2] - x[1:-1])
    )
    out[2::3] = (
        2. - numpy.cos(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        + numpy.sin(x[1:-1] - x[2:]) * numpy.cos(x[1:-1] + x[2:])
    )
    return out

eval_jac_g.sparsity_indices = (
    numpy.repeat(numpy.arange(nvar-2), 3),
    numpy.array([numpy.arange(nvar-2), numpy.arange(1, nvar-1), numpy.arange(2, nvar)]).T.flatten())
# [0, 0, 0, 1, 1, 1, ...]
# [0, 1, 2, 1, 2, 3, ...]

def eval_h(x, lagrange, obj_factor, out):
    out[-1] = 0.
    out[:-2:2] = obj_factor * (2. + 400. * (3. * x[:-1] * x[:-1] - x[1:]))
    out[:-4:2] -= lagrange * (2. + x[:-2]) * numpy.exp(x[:-2] - x[1:-1])
    out[2::2] += obj_factor * 200.
    out[2:-2:2] += lagrange * (
        18. * x[1:-1]
        - 2. * numpy.sin(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        + 2. * numpy.cos(x[1:-1] - x[2:]) * numpy.cos(x[1:-1] + x[2:])
        - x[:-2] * numpy.exp(x[:-2] - x[1:-1])
    )
    out[4::2] += lagrange * (
        -2. * numpy.sin(x[1:-1] - x[2:]) * numpy.sin(x[1:-1] + x[2:])
        - 2. * numpy.cos(x[1:-1] - x[2:]) * numpy.cos(x[1:-1] + x[2:])
    )
    out[1::2] = obj_factor * (-400. * x[:-1])
    out[1:-2:2] += lagrange * (1. + x[:-2]) * numpy.exp(x[:-2] - x[1:-1])
    return out


eval_h.sparsity_indices = (
    numpy.repeat(numpy.arange(nvar), 2)[:2*nvar-1],
    numpy.array([numpy.arange(nvar), numpy.arange(1, nvar+1)]).T.flatten()[:2*nvar-1]
)


nlp = ipyopt.Problem(nvar, x_L, x_U, ncon, g_L, g_U, eval_jac_g.sparsity_indices,
                     eval_h.sparsity_indices, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h)

print("Going to call solve")
print("x0 = {}".format(x0))
zl = numpy.zeros(nvar)
zu = numpy.zeros(nvar)
constraint_multipliers = numpy.zeros(ncon)
_x, obj, status = nlp.solve(x0, mult_g=constraint_multipliers,
                            mult_x_L=zl, mult_x_U=zu)


def print_variable(variable_name, value):
    for i, val in enumerate(value):
        print("{}[{}] = {}".format(variable_name, i, val))


print("Solution of the primal variables, x")
print_variable("x", _x)

print("Solution of the bound multipliers, z_L and z_U")
print_variable("z_L", zl)
print_variable("z_U", zu)

print("Solution of the constraint multipliers, lambda")
print_variable("lambda", constraint_multipliers)

print("Objective value")
print("f(x*) = {}".format(obj))
