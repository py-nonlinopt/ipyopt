"""Unittests"""
# pylint: disable=missing-function-docstring

import unittest
from unittest import mock
from typing import Any

import numpy
import ipyopt
import ipyopt.optimize

try:
    import scipy
    import scipy.optimize

    have_scipy = True
except ImportError:
    have_scipy = True
try:
    from . import c_capsules

    have_c_capsules = True
except ImportError:
    have_c_capsules = False


def e_x(n):
    """x unit vector"""
    out = numpy.zeros(n)
    out[0] = 1.0
    return out


def sparsity_g(n: int):
    return (
        numpy.zeros(n, dtype=int),
        numpy.arange(n, dtype=int),
    )


def sparsity_h(n: int):
    return (numpy.arange(n, dtype=int), numpy.arange(n, dtype=int))


def x_L(n: int) -> numpy.ndarray:
    return numpy.full((n,), -10.0)


def x_U(n: int) -> numpy.ndarray:
    return numpy.full((n,), 10.0)


def generic_problem(module, with_hess: bool = False, **kwargs):
    n = module.n
    eval_jac_g_sparsity_indices = sparsity_g(n)
    eval_h_sparsity_indices = sparsity_h(n)
    if with_hess:
        kwargs["eval_h"] = module.h
    _x_L = x_L(n)
    _x_U = x_U(n)

    g_L = numpy.array([0.0])
    g_U = numpy.array([4.0])

    p = ipyopt.Problem(
        n,
        _x_L,
        _x_U,
        1,
        g_L,
        g_U,
        eval_jac_g_sparsity_indices,
        eval_h_sparsity_indices,
        module.f,
        module.grad_f,
        module.g,
        module.jac_g,
        **kwargs
    )
    p.set(print_level=0, sb="yes")
    return p


def PyModule(_n, wrap_eval_h=lambda f: f):
    _e_x = e_x(_n)

    class _PyModule:
        """Set of pure python callbacks"""

        e_x = _e_x
        n = _n

        @staticmethod
        def f(x):
            return numpy.sum(x ** 2)

        @staticmethod
        def grad_f(x, out):
            out[()] = 2.0 * x
            return out

        @staticmethod
        def g(x, out):
            """Constraint function: squared distance to (1, 0, ..., 0)"""
            out[0] = numpy.sum((x - _e_x) ** 2)
            return out

        @staticmethod
        def jac_g(x, out):
            out[()] = 2.0 * (x - _e_x)
            return out

        @staticmethod
        @wrap_eval_h
        def h(_x, lagrange, obj_factor, out):
            out[()] = numpy.full((_n,), 2.0 * (obj_factor + lagrange[0]))
            return out

    return _PyModule


class Base:
    """Just a wrapper "namespace" to prevent discovery / running of the base test case"""

    class TestSimpleProblem(unittest.TestCase):
        """Base class for test suites for pure python callbacks, PyCapsule, scipy.LowLevelCallable"""

        function_set: Any = None

        def setUp(self):
            n = self.function_set.n
            self.x0 = numpy.full((n,), 0.1)
            self.zl = numpy.zeros(n)
            self.zu = numpy.zeros(n)
            self.constraint_multipliers = numpy.zeros(1)
            self.n = n

        def _solve(self, **kwargs):
            p = generic_problem(self.function_set, **kwargs)
            x, obj, status = p.solve(
                self.x0,
                mult_g=self.constraint_multipliers,
                mult_x_L=self.zl,
                mult_x_U=self.zu,
            )
            numpy.testing.assert_array_almost_equal(x, numpy.zeros(self.n))
            numpy.testing.assert_array_almost_equal(obj, 0.0)
            numpy.testing.assert_array_equal(status, 0)

        def test_optimize(self):
            n = self.function_set.n
            result = scipy.optimize.minimize(
                fun=self.function_set.f,
                x0=self.x0,
                method=ipyopt.optimize.ipopt,
                jac=ipyopt.optimize.JacEnvelope(self.function_set.grad_f),
                hess=self.function_set.h,
                bounds=list(zip(x_L(n), x_U(n))),
                constraints=ipyopt.optimize.Constraint(
                    fun=self.function_set.g,
                    jac=self.function_set.jac_g,
                    lb=numpy.array([0.0]),
                    ub=numpy.array([4.0]),
                    jac_sparsity_indices=sparsity_g(n),
                ),
                options=dict(
                    hess_sparsity_indices=sparsity_h(n),
                ),
            )
            numpy.testing.assert_array_almost_equal(result.x, numpy.zeros(self.n))
            numpy.testing.assert_array_almost_equal(result.fun, 0.0)
            numpy.testing.assert_array_equal(result.status, 0)
            numpy.testing.assert_array_equal(result.success, True)
            self.assertTrue(result.nfev > 0)
            self.assertTrue(result.njev > 0)
            self.assertTrue(result.nit > 0)


@unittest.skipIf(not have_c_capsules, "c_capsules not built")
class TestSimpleProblem(Base.TestSimpleProblem):
    """Test suite for PyCapsule"""

    function_set = c_capsules

    def setUp(self):
        super().setUp()
        c_capsules.capsule_set_context(c_capsules.h, None)
        c_capsules.capsule_set_context(c_capsules.intermediate_callback, None)

    def tearDown(self):
        c_capsules.capsule_set_context(c_capsules.h, None)
        c_capsules.capsule_set_context(c_capsules.intermediate_callback, None)

    def test_simple_problem(self):
        for with_hess in (True, False):
            h_callback = mock.Mock()
            c_capsules.capsule_set_context(c_capsules.h, h_callback)
            with self.subTest(with_hess=with_hess):
                self._solve(with_hess=with_hess)
                self.assertEqual(h_callback.called, with_hess)

    def test_simple_problem_with_intermediate_callback(self):
        callback = mock.Mock()
        c_capsules.capsule_set_context(c_capsules.intermediate_callback, callback)
        self._solve(intermediate_callback=c_capsules.intermediate_callback)
        callback.assert_called()


@unittest.skipIf(
    not have_c_capsules or not have_scipy,
    "c_capsules not built or scipy not available",
)
class TestSimpleProblemScipy(Base.TestSimpleProblem):
    """Test suite for scipy.LowLevelCallable"""

    def setUp(self):
        class ScipyModule:
            """Converts c_capsules into a set of scipy.LowLevelCallable"""

            n = c_capsules.n
            f = scipy.LowLevelCallable(c_capsules.f)
            grad_f = scipy.LowLevelCallable(c_capsules.grad_f)
            g = scipy.LowLevelCallable(c_capsules.g)
            jac_g = scipy.LowLevelCallable(c_capsules.jac_g)
            h = scipy.LowLevelCallable(c_capsules.h)
            intermediate_callback = scipy.LowLevelCallable(
                c_capsules.intermediate_callback
            )

        self.function_set = ScipyModule
        super().setUp()

    def test_simple_problem(self):
        for with_hess in (True, False):
            with self.subTest(with_hess=with_hess):
                self._solve(with_hess=with_hess)


class TestSimpleProblemPy(Base.TestSimpleProblem):
    """Test suite for pure python callbacks"""

    def setUp(self):
        self.function_set = PyModule(_n=4, wrap_eval_h=lambda f: mock.Mock(wraps=f))
        super().setUp()

    def test_simple_problem(self):
        for with_hess in (True, False):
            with self.subTest(with_hess=with_hess):
                self.function_set.h.reset_mock()
                self._solve(with_hess=with_hess)
                self.assertEqual(self.function_set.h.called, with_hess)


class TestIPyOpt(unittest.TestCase):
    """Test suite for problem scaling / intermediate_callback - pure python callbacks"""

    function_set = PyModule(_n=4)

    def test_problem_scaling(self):
        p = generic_problem(self.function_set)
        x0 = numpy.full((self.function_set.n,), 0.1)
        p.set(nlp_scaling_method="user-scaling")
        # Maximize instead of minimize:
        p.set_problem_scaling(obj_scaling=-1.0)
        x, obj, status = p.solve(x0)
        # -> Solution x should be the point within the circle
        # around e_x with radius 2 with the largest distance
        # to the origin, i.e. 3*e_x = (3,0,...,0)
        _e_x = e_x(self.function_set.n)
        numpy.testing.assert_array_almost_equal(x, 3.0 * _e_x)
        numpy.testing.assert_array_almost_equal(obj, 9.0)
        numpy.testing.assert_array_equal(status, 0)

    def test_problem_scaling_constructor(self):
        # Same again, but set scaling during problem creation
        p = generic_problem(self.function_set, obj_scaling=-1.0)
        x0 = numpy.full((self.function_set.n,), 0.1)
        p.set(nlp_scaling_method="user-scaling")
        x, obj, status = p.solve(x0)
        _e_x = e_x(self.function_set.n)
        numpy.testing.assert_array_almost_equal(x, 3.0 * _e_x)
        numpy.testing.assert_array_almost_equal(obj, 9.0)
        numpy.testing.assert_array_equal(status, 0)

    # @staticmethod
    # def test_problem_scaling_x():
    #    p = generic_problem(self.function_set)
    #    x0 = numpy.full((self.function_set.n,), 0.1)
    #    p.set(nlp_scaling_method="user-scaling")
    #    p.set(print_level=5)
    #    # Reflect x space:
    #    p.set_problem_scaling(obj_scaling=-1., x_scaling=numpy.full((n,), 1.), g_scaling = numpy.array([2.]))
    #    x, obj, status = p.solve(x0)
    #    # -> Solution x should be the point within the circle
    #    # around e_x with radius 2 with the largest distance
    #    # to the origin, i.e. 3*e_x = (3,0,...,0)
    #    numpy.testing.assert_array_almost_equal(x, -3.*self.function_set.e_x)
    #    numpy.testing.assert_array_almost_equal(obj, 9.)
    #    numpy.testing.assert_array_equal(status, 0)

    def test_intermediate_callback(self):
        x0 = numpy.full((self.function_set.n,), 0.1)
        intermediate_callback = mock.Mock(return_value=True)
        with self.subTest("Callback via constructor"):
            p = generic_problem(
                self.function_set, intermediate_callback=intermediate_callback
            )
            p.solve(x0)
            intermediate_callback.assert_called()
        with self.subTest("Callback not returning a bool"):
            intermediate_callback = mock.Mock()
            p = generic_problem(
                self.function_set, intermediate_callback=intermediate_callback
            )
            with self.assertRaises(RuntimeError):
                p.solve(x0)


@unittest.skipIf(not have_c_capsules, "c_capsules not built")
class TestIPyOptC(TestIPyOpt):
    """PyCapsule variant of TestIPyOpt"""

    function_set = c_capsules


class TestGetIpoptOptions(unittest.TestCase):
    """Tests for get_ipopt_options"""

    def test_get_ipopt_options(self):
        self.assertTrue(
            "print_level" in {opt["name"] for opt in ipyopt.get_ipopt_options()}
        )
