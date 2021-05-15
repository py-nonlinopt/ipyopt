scipy optimize method
=====================

ipyopt also comes with a ``ipopt`` method for
`scipy.optimize.minimize`_::

    result = scipy.optimize.minimize(
        fun=...,
        x0=...,
        jac=...,
        constraints=...,
        method=ipyopt.optimize.ipopt,
        ...
        )

.. warning::

   The ``ipopt`` method differs in some points from the standard scipy
   methods:

   * The argument ``jac`` is mandatory (explicitly use
     `scipy.optimize.approx_fprime`_ if you want to numerically
     approximate it)
   * ``hess`` is not the Hessian of the objective function ``f`` but the
     Hessian of the Lagrangian ``L(x) = obj_factor * f(x) + lagrange *
     g(x)``, where ``g`` is the constraint residuals.
   * The argument ``constraints`` is mandatory. It is also not a list
     as in usual scipy optimize methods, but a single
     :class:`Constraint` instance.

If you are looking for a solution whose API is closer to the usual
scipy interface, have a look at `cyipopt <https://github.com/mechmotum/cyipopt>`_.

Module optimize
---------------

.. automodule:: ipyopt.optimize
    :members:
    :noindex:

.. _`PyCapsule`: https://docs.python.org/3/c-api/capsule.html
.. _`scipy.LowLevelCallable`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html?highlight=lowlevelcallable#scipy.LowLevelCallable
