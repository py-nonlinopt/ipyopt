from typing import Any
from ipyopt.ipyopt import PyCapsule

n: int
m: int
f: PyCapsule
grad_f: PyCapsule
g: PyCapsule
jac_g: PyCapsule
h: PyCapsule
intermediate_callback: PyCapsule

def capsule_set_context(c: PyCapsule, ctx: Any) -> None: ...
def capsule_get_context(c: PyCapsule) -> Any: ...
