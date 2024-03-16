#!/bin/env python3

"""The same model as Ipopt/examples/hs071.

This example shows, how to use cython generated PyCapsule objects.
The definition of the capsules are in hs071_capsules.pyx and are
compiled via pyximport in this file.
"""

import numpy as np
from hs071_capsules import __pyx_capi__ as capsules

import ipyopt

nvar = 4
x_l = np.ones(nvar) * 1.0
x_u = np.ones(nvar) * 5.0

ncon = 2

g_l = np.array([25.0, 40.0])
g_u = np.array([2.0e19, 40.0])

# The following declares, that jac_g only has non zero
# entries at "*" (in this case no zeros):
# / * * * * \
# \ * * * * /
eval_jac_g_sparsity_indices = (
    np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    np.array([0, 1, 2, 3, 0, 1, 2, 3]),
)


# The following declares, that h only has non zero
# entries at "*":
# / * 0 0 0 \
# | * * 0 0 |
# | * * * 0 |
# \ * * * * /
eval_h_sparsity_indices = (
    np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
    np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]),
)


nlp = ipyopt.Problem(
    nvar,
    x_l,
    x_u,
    ncon,
    g_l,
    g_u,
    eval_jac_g_sparsity_indices,
    eval_h_sparsity_indices,
    capsules["f"],
    capsules["grad_f"],
    capsules["g"],
    capsules["jac_g"],
    capsules["h"],
)

x0 = np.array([1.0, 5.0, 5.0, 1.0])

print(f"Going to call solve with x0 = {x0}")
zl = np.zeros(nvar)
zu = np.zeros(nvar)
constraint_multipliers = np.zeros(ncon)
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
