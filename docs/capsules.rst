Using PyCapsules
================

Instead of passing pure Python callables to :class:`ipyopt.Problem`
which causes some overhead when calling the Python callables from C++,
we also can pass C callbacks encapsulated in `PyCapsule`_ objects.

.. warning:: **Segfaults**

     When working with C callbacks inside `PyCapsule`_ objects, ipyopt
     will always assume, that the C callbacks have the correct
     signature. If this is not the case, expect memory errors and thus crashes.


`PyCapsule`_ can be defined in C extensions. For an example on how to
do this from scratch, see `module.c
<https://gitlab.com/g-braeunlich/ipyopt/-/blob/master/test/c_capsules/src/module.c>`_.

A much more convenient way is to use `Cython`_. Then you don't need to
write boilerplate code to create a python module. All you still need
to do is to define the objective function, the constraint residuals
and their derivatives:

.. literalinclude:: ../examples/hs071_capsules.pyx
    :language: python

To just in time compile the ``pyx``, file, you can use Cython's
``pyximport``::

  import pyximport
  pyximport.install(language_level=3)


The compiled C extension can then be included. It will contain an
attribute ``__pyx_capi__`` containing the PyCapsules::

  from hs071_capsules import __pyx_capi__ as capsules

  nlp = ipyopt.Problem(
      ...
      capsules["f"],
      capsules["grad_f"],
      capsules["g"],
      capsules["jac_g"],
      capsules["h"],
  )


.. _`PyCapsule`: https://docs.python.org/3/c-api/capsule.html
.. _`Cython`: https://cython.org/
