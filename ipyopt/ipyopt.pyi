import sys
from typing import Any, Callable, NewType, Sequence, TypeAlias

import numpy as np
import scipy

if sys.version_info >= (3, 7):
    NDArrayF64: TypeAlias = np.typing.NDArray[np.float64]
else:
    NDArrayF64: TypeAlias = np.ndarray

PyCapsule = NewType("PyCapsule", object)

class Problem:
    stats: dict[str, int]

    def __init__(
        self,
        n: int,
        x_l: NDArrayF64,
        x_u: NDArrayF64,
        m: int,
        g_l: NDArrayF64,
        g_u: NDArrayF64,
        sparsity_indices_jac_g: tuple[
            Sequence[int] | NDArrayF64, Sequence[int] | NDArrayF64
        ],
        sparsity_indices_h: tuple[
            Sequence[int] | NDArrayF64, Sequence[int] | NDArrayF64
        ],
        eval_f: Callable[[NDArrayF64], float] | PyCapsule | scipy.LowLevelCallable,
        eval_grad_f: (
            Callable[[NDArrayF64, NDArrayF64], Any] | PyCapsule | scipy.LowLevelCallable
        ),
        eval_g: (
            Callable[[NDArrayF64, NDArrayF64], Any] | PyCapsule | scipy.LowLevelCallable
        ),
        eval_jac_g: (
            Callable[[NDArrayF64, NDArrayF64], Any] | PyCapsule | scipy.LowLevelCallable
        ),
        eval_h: (
            Callable[[NDArrayF64, NDArrayF64, float, NDArrayF64], Any]
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
        x_scaling: NDArrayF64 | None = None,
        g_scaling: NDArrayF64 | None = None,
        ipopt_options: dict[str, int | float | str] | None = None,
    ): ...
    def solve(
        self,
        x: NDArrayF64,
        *,
        mult_g: NDArrayF64 | None = None,
        mult_x_L: NDArrayF64 | None = None,  # noqa: N803
        mult_x_U: NDArrayF64 | None = None,  # noqa: N803
    ) -> tuple[NDArrayF64, float, int]: ...
    def set(self, **kwargs: str | float) -> None: ...
    def set_problem_scaling(
        self,
        obj_scaling: float,
        x_scaling: NDArrayF64 | None = None,
        g_scaling: NDArrayF64 | None = None,
    ) -> None: ...

def get_ipopt_options() -> list[dict[str, Any]]: ...
