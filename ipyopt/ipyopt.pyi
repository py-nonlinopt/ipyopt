from __future__ import annotations

import sys
from typing import Optional, Tuple, Sequence, Union, Callable, Dict, List, Any, NewType
import numpy
import scipy

if sys.version_info >= (3, 7):
    from numpy.typing import NDArray

    np_array = NDArray[numpy.float_]
else:
    np_array = numpy.ndarray

PyCapsule = NewType("PyCapsule", object)

class Problem:
    stats: Dict[str, int]

    def __init__(
        self,
        n: int,
        x_l: np_array,
        x_u: np_array,
        m: int,
        g_l: np_array,
        g_u: np_array,
        sparsity_indices_jac_g: Tuple[
            Union[Sequence[int], np_array], Union[Sequence[int], np_array]
        ],
        sparsity_indices_h: Tuple[
            Union[Sequence[int], np_array], Union[Sequence[int], np_array]
        ],
        eval_f: Union[Callable[[np_array], float], PyCapsule, scipy.LowLevelCallable],
        eval_grad_f: Union[
            Callable[[np_array, np_array], Any],
            PyCapsule,
            scipy.LowLevelCallable,
        ],
        eval_g: Union[
            Callable[[np_array, np_array], Any],
            PyCapsule,
            scipy.LowLevelCallable,
        ],
        eval_jac_g: Union[
            Callable[[np_array, np_array], Any],
            PyCapsule,
            scipy.LowLevelCallable,
        ],
        eval_h: Optional[
            Union[
                Callable[[np_array, np_array, float, np_array], Any],
                PyCapsule,
                scipy.LowLevelCallable,
            ]
        ] = None,
        intermediate_callback: Optional[
            Union[
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
                ],
                PyCapsule,
                scipy.LowLevelCallable,
            ]
        ] = None,
        obj_scaling: float = 1.0,
        x_scaling: Optional[np_array] = None,
        g_scaling: Optional[np_array] = None,
        ipopt_options: Optional[dict[str, Union[int, float, str]]] = None,
    ): ...
    def solve(
        self,
        x: np_array,
        *,
        mult_g: Optional[np_array] = None,
        mult_x_L: Optional[np_array] = None,
        mult_x_U: Optional[np_array] = None,
    ) -> Tuple[np_array, float, int]: ...
    def set(self, **kwargs: Union[str, float, int]) -> None: ...
    def set_problem_scaling(
        self,
        obj_scaling: float,
        x_scaling: Optional[np_array] = None,
        g_scaling: Optional[np_array] = None,
    ) -> None: ...

def get_ipopt_options() -> List[dict[str, Any]]: ...
