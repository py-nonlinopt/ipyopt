#!/bin/env python3
"""The same model as Ipopt/examples/hs071

This example shows, how to use auto differentiated, auto compiled PyCapsules from sympy expressions.
"""

import numpy

import ipyopt
from ipyopt.sym_compile import array_sym, SymNlp

n = 4
m = 2

x = array_sym("x", n)

f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
g = [x[0] * x[1] * x[2] * x[3], x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]]

c_api = SymNlp(f, g).compile()

nlp = ipyopt.Problem(
    n=n,
    x_l=numpy.ones(n),
    x_u=numpy.full(n, 5.0),
    m=m,
    g_l=numpy.array([25.0, 40.0]),
    g_u=numpy.array([2.0e19, 40.0]),
    **c_api,
)

x0 = numpy.array([1.0, 5.0, 5.0, 1.0])

print(f"Going to call solve with x0 = {x0}")
zl = numpy.zeros(n)
zu = numpy.zeros(n)
constraint_multipliers = numpy.zeros(m)
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
