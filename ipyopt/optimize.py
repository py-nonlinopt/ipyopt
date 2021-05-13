"""ipyopt.optimize: scipy method for scipy.optimize.minimize
"""

from typing import (
    Optional,
    Union,
    Callable,
    Tuple,
    Sequence,
    TypeVar,
    Generic,
    Any,
    NamedTuple,
)
import warnings

import numpy
from scipy.optimize import OptimizeResult
from .ipyopt import Problem, get_ipopt_options


class Constraint(NamedTuple):
    """Constraints definition to be passed to scipy.optimize.minimize as its `constraints` argument when using the ipopt method.
    The constraints are defined by
    lb <= fun(x) <= ub

    :param fun: The constraint function. Signature: fun(x: numpy.ndarray, out: numpy.ndarray) -> Any
    :param jac: Jacobian of fun. Signature: jac(x: numpy.ndarray, out: numpy.ndarray) -> Any
    :param lb: Lower bounds
    :param ub: Upper bounds
    :param jac_sparsity_indices: Sparsity structure of jac. Must be given in the form ((i[0], ..., i[m-1]), (j[0], ..., j[m-1])), where (i[k], j[k]) k=0,...,m-1 are the non zero entries of jac
    """

    fun: Callable[[numpy.ndarray, numpy.ndarray], Any]
    jac: Callable[[numpy.ndarray, numpy.ndarray], Any]
    lb: numpy.ndarray
    ub: numpy.ndarray
    jac_sparsity_indices: Optional[
        Tuple[Union[Sequence[int], numpy.ndarray], Union[Sequence[int], numpy.ndarray]]
    ] = None


T = TypeVar("T")


class JacEnvelope(Generic[T]):
    """A wrapper for PyCapsules / scipy.LowLevelCallable, so they can be passed as the jac argument to scipy.optimize.minimize.

    If the jac argument is not a Callable, then scipy will assume that it is a bool. It will be evaluated as bool and None will be passed to the method. To circumwent this, you will have to wrap your PyCapsules / scipy.LowLevelCallable s with this wrapper and pass it to scipy.optimize.minimize as the jac argument."""

    def __init__(self, inner: T):
        self.inner = inner

    def __call__(self) -> T:
        return self.inner


IPOPT_OPTION_KEYS = {opt["name"] for opt in get_ipopt_options()}

# From: Ipopt/IpReturnCodes_inc.h
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
    fun: Callable[[numpy.ndarray], float],
    x0: numpy.ndarray,
    args: tuple,
    *,
    jac: Union[Callable[[numpy.ndarray, numpy.ndarray], Any], JacEnvelope],
    hess: Optional[
        Callable[[numpy.ndarray, numpy.ndarray, float, numpy.ndarray], Any]
    ] = None,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    constraints: Constraint,
    tol: Optional[float] = None,
    callback: Optional[
        Callable[
            [int, int, float, float, float, float, float, float, float, float], Any
        ]
    ] = None,
    maxiter: Optional[int] = None,
    disp: bool = False,
    obj_scaling: float = 1.0,
    x_scaling: Optional[numpy.ndarray] = None,
    constraint_scaling: Optional[numpy.ndarray] = None,
    hess_sparsity_indices: Optional[
        Tuple[Union[Sequence[int], numpy.ndarray], Union[Sequence[int], numpy.ndarray]]
    ] = None,
    **kwargs,
) -> OptimizeResult:
    """IPOpt Method for scipy.optimize.minimize (to be used as 'method' argument)

    Standard arguments:
    :param fun:, :param x0: same as in scipy.optimize.minimize
    :param args: must be ()
    :param jac: Gradient of fun. If you want to pass a scipy.LowLevelCallable or a PyCapsule, you have to wrap it with :JacEnvelope: (see its documentation). In contrast to standard scipy.optimize.minimize this argument is mandatory. Use scipy.optimize.approx_fprime to numerically aproximate the derivative for pure python callables. This wont work for scipy.LowLevelCallable / PyCapsule.
    :param hess: Hessian of the Lagrangian L. hess(x, lag, obj_fac, out) should write into out the value of the Hessian of L = obj_fac*fun + <lag, constraint.fun>, where <.,.> denotes the euclidean inner product.
    :param bounds: Bounds for the x variable space
    :param constraints: See doc of :Constraint:
    :param tol: According to scipy.optimize.minimize
    :param callback: Will be called after each iteration. Must have the same signature as the intermediate_callback argument for :ipyopt.Problem:. See the IPOpt documentation for the meaning of the arguments.
    :param maxiter: According to scipy.optimize.minimize.
    :param disp: According to scipy.optimize.minimize.
    :param obj_scaling: Scaling factor for the objective value.
    :param x_scaling: Scaling factors for the x space.
    :param constraint_scaling: Scaling factors for the constraint space.
    :param hess_sparsity_indices: Sparsity indices for hess. Must be given in the form ((i[0], ..., i[n-1]), (j[0], ..., j[n-1])), where (i[k], j[k]) k=0,...,n-1 are the non zero entries of hess
    """
    if args:
        raise ValueError(
            "Passing arguments to function is not supported. Use closures or callable class instances to give the function access to some arguments."
        )
    options = {key: val for key, val in kwargs.items() if key in IPOPT_OPTION_KEYS}
    unsupported_args = frozenset(kwargs) - frozenset(options) - {"hessp"}
    if unsupported_args:
        warnings.warn(
            f"method ipopt: Got unsupported arguments: {', '.join(unsupported_args)}"
        )
    if not disp:
        options.setdefault("print_level", 0)
        options.setdefault("sb", "yes")
    if tol is not None:
        options["tol"] = tol
    if maxiter is not None:
        if "max_iter" in options:
            warnings.warn(
                "method ipopt: passed maxiter via argument 'max_iter' and 'maxiter'. Only 'maxiter' will be taken."
            )
        options["max_iter"] = maxiter
    n = x0.size
    m = constraints.lb.size
    if bounds is not None:
        x_l, x_u = numpy.array(bounds).T.copy()
    else:
        x_l = numpy.full(n, -float("inf"))
        x_u = numpy.full(n, float("inf"))
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
