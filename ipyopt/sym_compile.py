"""Automatic differentiation and code generation"""

import os
import sys
import importlib
from typing import Any, Union, List, Sequence, Tuple, Dict, Callable

from setuptools import Extension, Distribution
from sympy import Symbol, ccode, Expr
from sympy.codegen.ast import Assignment, CodeBlock
from sympy import S


class SymNlp:  # pylint: disable=too-many-instance-attributes
    """Symbolic definition of a NLP

    Only, sympy expressions for f, g needed, derivatives will be calculated automatically.
    The method :meth:`compile` will compile all expressions into PyCapsules and return them in a dict.
    """

    n: int
    m: int
    f: Expr
    grad_f: Sequence[Expr]
    g: Sequence[Expr]
    jac_g: Sequence[Expr]
    jac_g_sparsity_indices: Tuple[Sequence[int], Sequence[int]]
    h: Sequence[Expr]
    h_sparsity_indices: Tuple[Sequence[int], Sequence[int]]

    def __init__(self, f: Expr, g: Sequence[Expr]):
        self.f = f
        self.g = g

        x = sorted(f.atoms(), key=lambda s: str(s.name))
        self.m = m = len(g)
        self.n = len(x)

        l = array_sym("lambda", m)
        s = Symbol("obj_factor")

        L = s * f + sum(l_i * g_i for l_i, g_i in zip(l, g))
        """Lagrangian"""

        jac_g = [[g_i.diff(x_i) for x_i in x] for g_i in g]
        jac_g_sparse, jac_g_sparsity_indices = sparsify(jac_g)

        h = [[dL.diff(x_j) for x_j in x] for dL in [L.diff(x_i) for x_i in x]]
        h_sparse, h_sparsity_indices = sparsify(ll_triangular(h))
        self.grad_f = [f.diff(x_i) for x_i in x]
        self.jac_g = jac_g_sparse
        self.h = h_sparse
        self.h_sparsity_indices = transpose(h_sparsity_indices)
        self.jac_g_sparsity_indices = transpose(jac_g_sparsity_indices)

    def compile(self) -> Dict[str, Any]:
        """Compile all expressions into PyCapsules.

        Returns capsules in a dict with keys compatible with :class:`ipyopt.Problem`."""
        code = generate_c_code(self)
        c_api = compile_c(code)
        c_api.update(
            {
                "sparsity_indices_jac_g": self.jac_g_sparsity_indices,
                "sparsity_indices_h": self.h_sparsity_indices,
            }
        )
        return c_api


def array_sym(name: str, dim: int) -> List[Symbol]:
    """Creates a list of sympy symbols representing a vector valued symbol."""
    return [Symbol(f"{name}[{i}]") for i in range(dim)]


def c_function_body(expr: Sequence[Expr]) -> CodeBlock:
    """Prepares a sympy CodeBlock consisting of scalar assignment statements to store the expression in some target variable. This will be used to generate a C function body.

    :meta private:"""
    dim = len(expr)
    out = array_sym("out", dim)
    return CodeBlock(*(Assignment(out_i, expr_i) for out_i, expr_i in zip(out, expr)))


def sparsify(
    expr: Sequence[Sequence[Expr]],
) -> Tuple[Sequence[Expr], Sequence[Tuple[int, int]]]:
    """Flattens a matrix of sympy expressions, removes zeros

    Returns:
        The flattened, sparsified list of expression and the index pairs of non zero entries within the matrix.

    :meta private:"""
    indices, values = zip(
        *(
            ((i, j), val)
            for i, row in enumerate(expr)
            for j, val in enumerate(row)
            if val != S.Zero
        )
    )
    return values, indices


def ll_triangular(h: Sequence[Sequence[Expr]]) -> Sequence[Sequence[Expr]]:
    """Replaces the uper right triangular part (excluding the diagonal) of the input with zeros

    :meta private:"""
    return [
        [h_ij if j <= i else S.Zero for j, h_ij in enumerate(h_i)]
        for i, h_i in enumerate(h)
    ]


def transpose(
    index_pairs: Sequence[Tuple[int, int]]
) -> Tuple[Sequence[int], Sequence[int]]:
    """Turns a sequence of index pairs (i,j) into
    a sequence of i indices and a sequence of j indeces.

    :meta private:"""
    i, j = zip(*index_pairs)
    return (i, j)


def generate_c_code(nlp: SymNlp) -> str:
    """Writes C code for all symbolic expressions of a :class:`SymNlp`

    :meta private:"""
    grad_f_codeblock = c_function_body(nlp.grad_f)
    g_codeblock = c_function_body(nlp.g)
    jac_g_codeblock = c_function_body(nlp.jac_g)
    h_codeblock = c_function_body(nlp.h)

    # Just to give type info to ccode:
    c_code: Callable[[Union[CodeBlock, Expr]], str] = ccode

    return f"""
#include "Python.h"
#include <stdbool.h>

#define N {nlp.n}
#define M {nlp.m}

static bool f(int n, const double *x, double *obj_value, void *userdata) {{
  *obj_value = { c_code(nlp.f) };
  return true;
}}

static bool grad_f(int n, const double *x, double *out, void *userdata) {{
  { c_code(grad_f_codeblock) }
  return true;
}}

static bool g(int n, const double *x, int m, double *out, void *userdata) {{
  { c_code(g_codeblock) }
  return true;
}}

static bool jac_g(int n, const double *x, int m, int n_out, double *out,
                  void *userdata) {{
  { c_code(jac_g_codeblock) }
  return true;
}}

static bool h(int n, const double *x, double obj_factor, int m,
              const double *lambda, int n_out, double *out, void *userdata) {{
  { c_code(h_codeblock) }
  return true;
}}

static struct PyModuleDef moduledef = {{
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "c_api",
    .m_doc = "C PyCapsules",
    .m_size = -1,
    .m_methods = NULL,
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL,
}};

PyMODINIT_FUNC PyInit_c_api(void) {{
  PyObject *module = PyModule_Create(&moduledef);
  if (module == NULL)
    return NULL;

  PyObject *py_c_api = PyDict_New();
  PyDict_SetItemString(py_c_api, "eval_f", PyCapsule_New((void *)f, "f", NULL));
  PyDict_SetItemString(py_c_api, "eval_grad_f", PyCapsule_New((void *)grad_f, "grad_f", NULL));
  PyDict_SetItemString(py_c_api, "eval_g", PyCapsule_New((void *)g, "g", NULL));
  PyDict_SetItemString(py_c_api, "eval_jac_g", PyCapsule_New((void *)jac_g, "jac_g", NULL));
  PyDict_SetItemString(py_c_api, "eval_h", PyCapsule_New((void *)h, "h", NULL));
  if (PyModule_AddObject(module, "__capi__", py_c_api) < 0 ||
      PyModule_AddIntConstant(module, "n", N) < 0 ||
      PyModule_AddIntConstant(module, "m", M) < 0) {{
    Py_XDECREF(py_c_api);
    Py_DECREF(module);
    return NULL;
  }}

  return module;
}}
"""


def compile_c(code: str) -> Dict[str, Any]:
    """Compiles C code into a C extension, loads the extension and returns the included dict of compiled PyCapsules.

    This assumes that the code defines a module member
    ``__capi__``.

    :meta private:"""
    build_dir = "build"
    c_file_path = os.path.join(build_dir, "src", "module.c")
    prepare_c_src(c_file_path, code)
    dist = Distribution(
        {
            "name": "c_api",
            "ext_modules": [Extension("c_api", sources=[c_file_path])],
            "script_name": "setup.py",
            "script_args": ["build_ext", "--inplace"],
        }
    )
    dist.parse_command_line()
    dist.run_commands()
    sys.path.append(os.getcwd())
    return importlib.import_module("c_api").__capi__  # type: ignore


def prepare_c_src(path: str, code: str) -> None:
    """Writes code into a file, if code differs from file contents, or if file does not exist.
    Creates parent directories if they dont exist.

    :meta private:"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == code:
                return
    except FileNotFoundError:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
