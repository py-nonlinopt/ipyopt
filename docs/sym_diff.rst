Automatic symbolic derivatives / code generation
================================================

Probably the most convenient way to use ipyopt is to let sympy compute
your derivatives automatically. This however is only possible out of
the box for simple functions without exotic matrix operations /
special functions (i.e. Bessel functions).

At this stage, ipyopt only ships a proof of concept to show that this
is possible. Therefore the module :mod:`ipyopt.sym_compile` is to be considered experimental.

Its ``array_sym`` function can be used to declare symbolic vector
valued variables to be used in the objective function ``f`` and the
constraint residuals ``g``. You then define expressions for ``f`` and
``g``. Currently the variable used in this expression has to be
exactly ``x``. Choosing a different letter wont work in the current
version.
The expressions for ``f`` and ``g`` are sufficient. :mod:`ipyopt.sym_compile`
will then be able to

* automatically build derivatives
* generate C code for all expressions
* compile the C code to a python extension
* load the python extension and return the contained PyCapsule objects in a dict.

The returned dict can directly be passed to ``ipyopt.Problem``.

Here is Ipopt's ``hs071`` example reimplemented using this approach:

.. literalinclude:: ../examples/hs071_sym.py
    :start-at: import numpy
    :end-before: x0 =

Module sym_compile
------------------

.. automodule:: ipyopt.sym_compile
    :members:
    :noindex:
