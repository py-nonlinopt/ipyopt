"""ipyopt.optimize: ipopt method for `scipy.optimize.minimize`_.

.. _`scipy.optimize.minimize`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
"""

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NamedTuple,
    TypeVar,
)

import numpy as np
from scipy.optimize import OptimizeResult

from .ipyopt import Problem, get_ipopt_options

if TYPE_CHECKING:  # Only processed by mypy
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray


class Constraint(NamedTuple):
    """Constraints definition.

    To be passed to `scipy.optimize.minimize`_
    as its ``constraints`` argument when using the ipopt method.

    The constraints are defined by::

        lb <= fun(x) <= ub
    """

    fun: "Callable[[NDArray[np.float64], NDArray[np.float64]], Any]"
    """Constraint function.

    Signature is ``fun(x: NDArray[np.float64], out: NDArray[np.float64]) -> Any``"""
    jac: "Callable[[NDArray[np.float64], NDArray[np.float64]], Any]"
    """Jacobian of ``fun``.

    Signature is ``jac(x: NDArray[np.float64], out: NDArray[np.float64]) -> Any``"""
    lb: "NDArray[np.float64]"
    """Lower bounds"""
    ub: "NDArray[np.float64]"
    """Upper bounds"""
    jac_sparsity_indices: "tuple[Sequence[int] | NDArray[np.int64], Sequence[int] | NDArray[np.int64]] | None" = None
    """Sparsity structure of ``jac``.

    Must be given in the form ``((i[0], ..., i[m-1]), (j[0], ..., j[m-1]))``,
    where ``(i[k], j[k]), k=0,...,m-1`` are the non zero entries of ``jac``"""


T = TypeVar("T")


class JacEnvelope(Generic[T]):
    """A wrapper for `PyCapsule`_ / `scipy.LowLevelCallable`_ objects.

    This allows those kind of objects to passed as the ``jac`` argument
    of `scipy.optimize.minimize`_.

    If the ``jac`` argument is not callable, then `scipy.optimize.minimize`_
    will assume that it is a ``bool``. It will be evaluated to a ``bool`` and
    ``None`` will be passed to the method. To circumwent this, wrap
    your `PyCapsule`_ / `scipy.LowLevelCallable`_ objects with this wrapper and pass it
    to `scipy.optimize.minimize`_ as the ``jac`` argument.

    .. _PyCapsule: https://docs.python.org/3/c-api/capsule.html
    .. _scipy.LowLevelCallable: https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html
    """

    def __init__(self, inner: T):
        self.inner = inner

    def __call__(self) -> T:
        """Make the envelope callable."""
        return self.inner


IPOPT_OPTION_KEYS = {opt["name"] for opt in get_ipopt_options()}

# Source: `Ipopt/IpReturnCodes_inc.h`
IPOPT_RETURN_CODES = {
    0: "Solve Succeeded",
    1: "Solved to acceptable level",
    2: "Infeasible problem detected",
    3: "Search direction becomes too small",
    4: "Diverging Iterates",
    5: "User requested stop",
    6: "Feasible point found",
    -1: "Maximum iterations exceeded",
    -2: "Restoration failed",
    -3: "Error in step computation",
    -4: "Maximum CPU time exceeded",
    -10: "Not enough degrees of freedom",
    -11: "Invalid problem definition",
    -12: "Invalid option",
    -13: "Invalid number detected",
    -100: "Unrecoverable exception",
    -101: "NonIpopt exception thrown",
    -102: "Insufficient memory",
    -199: "Internal error",
}


def ipopt(
    fun: "Callable[[NDArray[np.float64]], float]",
    x0: "NDArray[np.float64]",
    args: "tuple[()]",
    *,
    jac: "Callable[[NDArray[np.float64], NDArray[np.float64]], Any] | JacEnvelope[Any]",
    hess: "Callable[[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]], Any] | None" = None,
    bounds: "Sequence[tuple[float, float]] | None" = None,
    constraints: Constraint,
    tol: "float | None" = None,
    callback: "Callable[[int, int, float, float, float, float, float, float, float, float], Any] | None" = None,
    maxiter: "int | None" = None,
    disp: bool = False,
    obj_scaling: float = 1.0,
    x_scaling: "NDArray[np.float64] | None" = None,
    constraint_scaling: "NDArray[np.float64] | None" = None,
    hess_sparsity_indices: "tuple[Sequence[int] | NDArray[np.int64], Sequence[int] | NDArray[np.int64]] | None" = None,
    **kwargs: Any,
) -> OptimizeResult:
    """Ipopt Method for `scipy.optimize.minimize`_ (to be used as ``method`` argument).

    Args:
        fun: Function to optimize.
        x0: same as in `scipy.optimize.minimize`_
        args: must be ``()``
        jac: Gradient of ``fun``.
          If you want to pass a `scipy.LowLevelCallable`_ or a
          `PyCapsule`_, you have to wrap it with :class:`JacEnvelope`
          (see its documentation). In contrast to standard
          `scipy.optimize.minimize`_ this argument is mandatory.
          Use `scipy.optimize.approx_fprime`_ to numerically approximate
          the derivative for pure python callables.
          This wont work for `scipy.LowLevelCallable`_ / `PyCapsule`_.
        hess: Hessian of the Lagrangian L.
          ``hess(x, lag, obj_fac, out)`` should write into ``out`` the
          value of the Hessian of::

                L = obj_fac*fun + <lag, constraint.fun>,

          where ``<.,.>`` denotes the euclidean inner product.
        bounds: Bounds for the x variable space
        constraints: See doc of :class:`Constraint`
        tol: According to `scipy.optimize.minimize`_
        callback: Will be called after each iteration.
          Must have the same signature as the ``intermediate_callback``
          argument for ``ipyopt.Problem``.
          See the Ipopt documentation for the meaning of the arguments.
        maxiter: According to `scipy.optimize.minimize`_.
        disp: According to `scipy.optimize.minimize`_.
        obj_scaling: Scaling factor for the objective value.
        x_scaling: Scaling factors for the x space.
        constraint_scaling: Scaling factors for the constraint space.
        hess_sparsity_indices: Sparsity indices for ``hess``.
          Must be given in the form ``((i[0], ..., i[n-1]), (j[0], ..., j[n-1]))``,
          where ``(i[k], j[k]), k=0,...,n-1`` are the non zero entries of ``hess``.
        kwargs: Options which will be forwarded to the call of IPOpt.

    Returns:
        An `scipy.optimize.OptimizeResult`_ instance

    .. _`scipy.optimize.approx_fprime`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html?highlight=approx_fprime#scipy.optimize.approx_fprime
    .. _`scipy.optimize.OptimizeResult`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html?highlight=optimizeresult#scipy.optimize.OptimizeResult
    """
    # pylint: disable=too-many-locals
    if args:
        msg = (
            "Passing arguments to function is not supported. "
            "Use closures or callable class instances to give the "
            "function access to some arguments."
        )
        raise ValueError(msg)
    options = {key: val for key, val in kwargs.items() if key in IPOPT_OPTION_KEYS}
    unsupported_args = frozenset(kwargs) - frozenset(options) - {"hessp"}
    if unsupported_args:
        warnings.warn(
            f"method ipopt: Got unsupported arguments: {', '.join(unsupported_args)}",
            stacklevel=1,
        )
    if not disp:
        options.setdefault("print_level", 0)
        options.setdefault("sb", "yes")
    if tol is not None:
        options["tol"] = tol
    if maxiter is not None:
        if "max_iter" in options:
            warnings.warn(
                "method ipopt: passed maxiter via argument 'max_iter' and 'maxiter'. "
                "Only 'maxiter' will be taken.",
                stacklevel=1,
            )
        options["max_iter"] = maxiter
    n = x0.size
    m = constraints.lb.size
    if bounds is not None:
        x_l, x_u = np.array(bounds).T.copy()
    else:
        x_l = np.full(n, -float("inf"))
        x_u = np.full(n, float("inf"))
    if isinstance(jac, JacEnvelope):
        jac = jac()
    p = Problem(
        n=n,
        x_l=x_l,
        x_u=x_u,
        m=m,
        g_l=constraints.lb,
        g_u=constraints.ub,
        sparsity_indices_jac_g=constraints.jac_sparsity_indices
        or (
            sum(((i,) * n for i in range(m)), ()),
            m * tuple(range(n)),
        ),
        sparsity_indices_h=hess_sparsity_indices
        or (
            sum(((i,) * n for i in range(n)), ()),
            n * tuple(range(n)),
        ),
        eval_f=fun,
        eval_grad_f=jac,
        eval_g=constraints.fun,
        eval_jac_g=constraints.jac,
        eval_h=hess,
        intermediate_callback=callback,
        obj_scaling=obj_scaling,
        x_scaling=x_scaling,
        g_scaling=constraint_scaling,
        ipopt_options=options,
    )
    x, obj_val, status = p.solve(x0)
    stats = p.stats
    return OptimizeResult(
        x=x,
        success=status == 0,
        status=status,
        message=IPOPT_RETURN_CODES[status],
        fun=obj_val,
        nfev=stats["n_eval_f"],
        njev=stats["n_eval_grad_f"],
        nit=stats["n_iter"],
    )
