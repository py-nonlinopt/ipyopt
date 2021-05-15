from typing import Optional, Tuple, Sequence, Union, Callable, Dict, List, Any, NewType
import numpy
import scipy

PyCapsule = NewType("PyCapsule", object)

class Problem:
    stats: Dict[str, int]
    def __init__(
        self,
        n: int,
        x_l: numpy.ndarray,
        x_u: numpy.ndarray,
        m: int,
        g_l: numpy.ndarray,
        g_u: numpy.ndarray,
        sparsity_indices_jac_g: Tuple[
            Union[Sequence[int], numpy.ndarray], Union[Sequence[int], numpy.ndarray]
        ],
        sparsity_indices_h: Tuple[
            Union[Sequence[int], numpy.ndarray], Union[Sequence[int], numpy.ndarray]
        ],
        eval_f: Union[
            Callable[[numpy.ndarray], float], PyCapsule, scipy.LowLevelCallable
        ],
        eval_grad_f: Union[
            Callable[[numpy.ndarray, numpy.ndarray], Any],
            PyCapsule,
            scipy.LowLevelCallable,
        ],
        eval_g: Union[
            Callable[[numpy.ndarray, numpy.ndarray], Any],
            PyCapsule,
            scipy.LowLevelCallable,
        ],
        eval_jac_g: Union[
            Callable[[numpy.ndarray, numpy.ndarray], Any],
            PyCapsule,
            scipy.LowLevelCallable,
        ],
        eval_h: Optional[
            Union[
                Callable[[numpy.ndarray, numpy.ndarray, float, numpy.ndarray], Any],
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
        x_scaling: Optional[numpy.ndarray] = None,
        g_scaling: Optional[numpy.ndarray] = None,
        ipopt_options: Optional[dict[str, Union[int, float, str]]] = None,
    ): ...
    def solve(
        self,
        x: numpy.ndarray,
        *,
        mult_g: Optional[numpy.ndarray] = None,
        mult_x_L: Optional[numpy.ndarray] = None,
        mult_x_U: Optional[numpy.ndarray] = None
    ) -> Tuple[numpy.ndarray, float, int]: ...
    def set(self, **kwargs): ...
    def set_problem_scaling(
        self,
        obj_scaling: float,
        x_scaling: Optional[numpy.ndarray] = None,
        g_scaling: Optional[numpy.ndarray] = None,
    ): ...

def get_ipopt_options() -> List[dict[str, Any]]: ...
