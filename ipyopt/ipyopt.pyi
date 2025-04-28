from collections.abc import Callable, Sequence
from typing import Any, NewType

import numpy as np
import scipy
from numpy.typing import NDArray

PyCapsule = NewType("PyCapsule", object)

class Problem:
    stats: dict[str, int]

    def __init__(
        self,
        n: int,
        x_l: NDArray[np.float64],
        x_u: NDArray[np.float64],
        m: int,
        g_l: NDArray[np.float64],
        g_u: NDArray[np.float64],
        sparsity_indices_jac_g: tuple[
            Sequence[int] | NDArray[np.int64], Sequence[int] | NDArray[np.int64]
        ],
        sparsity_indices_h: tuple[
            Sequence[int] | NDArray[np.int64], Sequence[int] | NDArray[np.int64]
        ],
        eval_f: Callable[[NDArray[np.float64]], float]
        | PyCapsule
        | scipy.LowLevelCallable,
        eval_grad_f: (
            Callable[[NDArray[np.float64], NDArray[np.float64]], Any]
            | PyCapsule
            | scipy.LowLevelCallable
        ),
        eval_g: (
            Callable[[NDArray[np.float64], NDArray[np.float64]], Any]
            | PyCapsule
            | scipy.LowLevelCallable
        ),
        eval_jac_g: (
            Callable[[NDArray[np.float64], NDArray[np.float64]], Any]
            | PyCapsule
            | scipy.LowLevelCallable
        ),
        eval_h: (
            Callable[
                [NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]],
                Any,
            ]
            | PyCapsule
            | scipy.LowLevelCallable
            | None
        ) = None,
        intermediate_callback: (
            Callable[
                [
                    int,
                    int,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    int,
                ],
                Any,
            ]
            | PyCapsule
            | scipy.LowLevelCallable
            | None
        ) = None,
        obj_scaling: float = 1.0,
        x_scaling: NDArray[np.float64] | None = None,
        g_scaling: NDArray[np.float64] | None = None,
        ipopt_options: dict[str, int | float | str] | None = None,
    ): ...
    def solve(
        self,
        x0: NDArray[np.float64],
        *,
        mult_g: NDArray[np.float64] | None = None,
        mult_x_L: NDArray[np.float64] | None = None,  # noqa: N803
        mult_x_U: NDArray[np.float64] | None = None,  # noqa: N803
    ) -> tuple[NDArray[np.float64], float, int]: ...
    def set(self, **kwargs: str | float) -> None: ...
    def set_problem_scaling(
        self,
        obj_scaling: float,
        x_scaling: NDArray[np.float64] | None = None,
        g_scaling: NDArray[np.float64] | None = None,
    ) -> None: ...

def get_ipopt_options() -> list[dict[str, Any]]: ...
