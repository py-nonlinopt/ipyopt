"""Solve a tiny NLP with full Ipopt output.

Run by cibuildwheel's test-command before the unit tests (see pyproject.toml).
The unit tests only assert on results; this script puts Ipopt's own
diagnostics (version banner, linear solver, iteration log) into the CI log
and fails if the trivial problem does not converge — turning "Ipopt silently
returns x0" failure modes into an actionable log.
"""

import numpy as np

import ipyopt

n = 2


def eval_f(x):
    """Objective: squared distance from the origin."""
    return float((x**2).sum())


def eval_grad_f(x, out):
    """Gradient of the objective."""
    out[:] = 2.0 * x
    return out


def eval_g(x, out):
    """Constraint: sum(x), bounded to [-10, 10]."""
    out[0] = x.sum()
    return out


def eval_jac_g(_x, out):
    """Jacobian of the constraint (dense row)."""
    out[:] = 1.0
    return out


problem = ipyopt.Problem(
    n,
    np.full(n, -10.0),
    np.full(n, 10.0),
    1,
    np.array([-10.0]),
    np.array([10.0]),
    (np.zeros(n, dtype=np.int64), np.arange(n, dtype=np.int64)),
    (np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64)),
    eval_f,
    eval_grad_f,
    eval_g,
    eval_jac_g,
)
problem.set(print_level=5)
x, obj, status = problem.solve(np.full(n, 0.1))
print(f"smoke test: status={status}, x={x}, obj={obj}")

if status != 0 or not np.allclose(x, 0.0, atol=1e-6):
    msg = f"Ipopt smoke test failed: status={status}, x={x}"
    raise SystemExit(msg)
