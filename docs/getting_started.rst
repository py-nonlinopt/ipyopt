Getting started
===============

Ipopt solves the following class of nonlinear programming problems (NLP):

.. math::

    \mathrm{min}_{\boldsymbol{x} \in \mathbb{R}^n} f(\boldsymbol{x}) \\
    \boldsymbol{g}_l \leq \boldsymbol{g}(\boldsymbol{x}) \leq
    \boldsymbol{g}_u \in \mathbb{R}^m \\
    \boldsymbol{x}_l \leq \boldsymbol{x} \leq
    \boldsymbol{x}_u \in \mathbb{R}^n
    
Lets start with a minimal example:

.. math::

    \boldsymbol{x} \in [-10, 10]^3 \subset \mathbb{R}^3 \\
    f(\boldsymbol{x}) = x_0^2 + x_1^2 + x_2^2 \\
    0 \leq g(\boldsymbol{x}) = (x_0 -1)^2 + x_1^2 + x_2^2 \leq 4 \in \mathbb{R}^1.

That is, :math:`\boldsymbol{x}_l = (-10, -10, -10)`,
:math:`\boldsymbol{x}_u = (10, 10, 10)`.

**Step 1:** Define the problem in Python

We first define the corresponding Python function for ``f`` and its derivative:

.. literalinclude:: ../test/test_ipyopt.py
    :dedent: 8
    :start-at: def f(
    :end-at: return

.. literalinclude:: ../test/test_ipyopt.py
    :dedent: 8
    :start-at: def grad_f(
    :end-at: return

.. note::

   For performance reasons, you have to write the value of
   ``grad_f`` into the ``out`` argument of the function (this way we can
   avoid unnecessary memory allocation). 

Next, we define ``g`` and its derivative:

.. literalinclude:: ../test/test_ipyopt.py
    :dedent: 8
    :start-at: def g(
    :end-at: return

.. literalinclude:: ../test/test_ipyopt.py
    :dedent: 8
    :start-at: def jac_g(
    :end-at: return

Here, The Jacobian :math:`\mathrm{D} g = 2 (x_0 - 1, x_1, x_2)`, thus the non
zero slots (all slots in this case) are ``(i,j) = (0,0), (0,1), (0,2)``.
Translated into python code::

    (numpy.array([0, 0, 0]), numpy.array([0, 1, 2]))

.. note::

    While in the notation ``(i,j) = (0,0), (0,1), (0,2)``, we use index pairs,
    you have to provide the collections of all ``i`` components
    (first tuple field) and the collection of all ``j`` components (second
    tuple field) as sparsity info arguments to ipyopt.

For maximal performance, we also can provide the Hessian of the
Lagrangian :math:`L(x) = \sigma f(x) + \langle \lambda,  g(x)\rangle`,
where :math:`\langle \cdot, \cdot \rangle` denotes the standard scalar
product (in this case just the usual multiplication, as we are in
:math:`\mathbb{R}^1`).
In our case:

.. math::

    \mathrm{Hess}\, L = \sigma \left( \begin{array}{lll}
    2 & 0 & 0 \\
    0 & 2 & 0 \\
    0 & 0 & 2
    \end{array}
    \right)
    + \lambda_0 \left( \begin{array}{lll}
      2 & 0 & 0 \\
      0 & 2 & 0 \\
      0 & 0 & 2
    \end{array} \right)
    = 2(\sigma + \lambda_0) \left(\begin{array}{lll}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1 \end{array} \right).

Therefore, we have nonzero entries at the diagonal, i.e. ``(i,j) =
(0,0), (1,1), (2,2)``.
The corresponding sparsity info argument is

``(numpy.array([0, 1, 2]), numpy.array([0, 1, 2]))`` and this is how
to compute the non zero entries:

.. literalinclude:: ../test/test_ipyopt.py
    :dedent: 8
    :start-at: def h(
    :end-at: return

Here, ``lagrange`` corresponds to :math:`\lambda` and ``obj_factor`` to
:math:`\sigma`.
If we don't provide the Hessian, Ipopt will numerically approximate it
for us, at the price of some performance loss.

Now, we are ready to define the Python problem:

.. code::

    import ipyopt
    nlp = ipyopt.Problem(
        n=2,
        x_l=numpy.array([-10.0, -10.0, -10.0]),
        x_u=numpy.array([10.0, 10.0, 10.]),
        m=1,
        g_l=numpy.array([0.0]),
        g_u=numpy.array([4.0]),
        sparsity_indices_jac_g=(numpy.array([0, 0, 0]), numpy.array([0, 1, 2])),
        sparsity_indices_h=(numpy.array([0, 1, 2]), numpy.array([0, 1, 2])),
        f,
        grad_f,
        g,
        jac_g,
        h
    )

**Step 2:** Solve the problem

We will use :math:`x_0 = (0.1, 0.1, 0.1)` as initial guess.

.. code::

    x, obj, status = nlp.solve(x0=numpy.array([0.1, 0.1, 0.1]))

As a result, we should obtain the solution ``x = (0.,0.,0.)``, ``obj =
f(x) = 0.`` and ``status = 0`` (meaning, that Ipopt found the optimal
solution within tolerance).
