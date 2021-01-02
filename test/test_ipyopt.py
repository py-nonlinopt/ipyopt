import unittest
from unittest import mock
import numpy
import ipyopt

def e_x(n):
    """x unit vector"""
    out = numpy.zeros(n)
    out[0] = 1.
    return out

def problem(n: int, with_hess: bool = False,
            wrap_eval_h=lambda f: f, **kwargs):
    _e_x = e_x(n)
    def eval_f(x):
        return numpy.sum(x**2)


    def eval_grad_f(x, out):
        out[()] = 2.*x
        return out


    def eval_g(x, out):
        """Constraint function: squared distance to (1, 0, ..., 0)"""
        out[0] = numpy.sum((x - _e_x)**2)
        return out


    def eval_jac_g(x, out):
        out[()] = 2.*(x - _e_x)
        return out

    eval_jac_g_sparsity_indices = (
        numpy.zeros(n, dtype=int),
        numpy.arange(n, dtype=int)
    )

    @wrap_eval_h
    def eval_h(_x, lagrange, obj_factor, out):
        out[()] = numpy.full((n,), 2.*(obj_factor + lagrange[0]))
        return out


    eval_h_sparsity_indices = (
        numpy.arange(n, dtype=int),
        numpy.arange(n, dtype=int)
    )
    if with_hess:
        kwargs["eval_h"] = eval_h
    x_L = numpy.full((n,), -10.)
    x_U = numpy.full((n,), 10.)

    g_L = numpy.array([0.0])
    g_U = numpy.array([4.0])

    p = ipyopt.Problem(n, x_L, x_U, 1, g_L, g_U,
                       eval_jac_g_sparsity_indices,
                       eval_h_sparsity_indices, eval_f,
                       eval_grad_f, eval_g, eval_jac_g,
                       **kwargs)
    p.set(print_level=0, sb="yes")
    return p


class TestIPyOpt(unittest.TestCase):
    def test_simple_problem(self):
        n = 4
        x0 = numpy.full((n,), 0.1)
        zl = numpy.zeros(n)
        zu = numpy.zeros(n)
        constraint_multipliers = numpy.zeros(1)
        class CallChecker:
            def __init__(self):
                self.called = False
            def __call__(self, f):
                def wrapped(*args, **kwargs):
                    self.called = True
                    return f(*args, **kwargs)
                return wrapped
        for with_hess in (False, True):
            with self.subTest(with_hess=with_hess):
                eval_h_call_checker = CallChecker()
                p = problem(n, with_hess=with_hess,
                            wrap_eval_h=eval_h_call_checker)
                x, obj, status = p.solve(x0, mult_g=constraint_multipliers,
                                         mult_x_L=zl, mult_x_U=zu)
                self.assertEqual(eval_h_call_checker.called, with_hess)
                numpy.testing.assert_array_almost_equal(x, numpy.zeros(n))
                numpy.testing.assert_array_almost_equal(obj, 0.)
                numpy.testing.assert_array_equal(status, 0)

    @staticmethod
    def test_problem_scaling():
        n = 4
        p = problem(n)
        x0 = numpy.full((n,), 0.1)
        p.set(nlp_scaling_method="user-scaling")
        # Maximize instead of minimize:
        p.set_problem_scaling(obj_scaling=-1.)
        x, obj, status = p.solve(x0)
        # -> Solution x should be the point within the circle
        # around e_x with radius 2 with the largest distance
        # to the origin, i.e. 3*e_x = (3,0,...,0)
        numpy.testing.assert_array_almost_equal(x, 3.*e_x(n))
        numpy.testing.assert_array_almost_equal(obj, 9.)
        numpy.testing.assert_array_equal(status, 0)

    @staticmethod
    def test_problem_scaling_constructor():
        # Same again, but set scaling during problem creation
        n = 4
        p = problem(n, obj_scaling=-1.)
        x0 = numpy.full((n,), 0.1)
        p.set(nlp_scaling_method="user-scaling")
        x, obj, status = p.solve(x0)
        numpy.testing.assert_array_almost_equal(x, 3.*e_x(n))
        numpy.testing.assert_array_almost_equal(obj, 9.)
        numpy.testing.assert_array_equal(status, 0)

    #@staticmethod
    #def test_problem_scaling_x():
    #    n = 4
    #    p = problem(n)
    #    x0 = numpy.full((n,), 0.1)
    #    p.set(nlp_scaling_method="user-scaling")
    #    p.set(print_level=5)
    #    # Reflect x space:
    #    p.set_problem_scaling(obj_scaling=-1., x_scaling=numpy.full((n,), 1.), g_scaling = numpy.array([2.]))
    #    x, obj, status = p.solve(x0)
    #    # -> Solution x should be the point within the circle
    #    # around e_x with radius 2 with the largest distance
    #    # to the origin, i.e. 3*e_x = (3,0,...,0)
    #    numpy.testing.assert_array_almost_equal(x, -3.*e_x(n))
    #    numpy.testing.assert_array_almost_equal(obj, 9.)
    #    numpy.testing.assert_array_equal(status, 0)

    def test_intermediate_callback(self):
        n = 4
        x0 = numpy.full((n,), 0.1)
        intermediate_callback = mock.Mock(return_value=True)
        with self.subTest("Callback via set_intermediate_callback"):
            p = problem(n)
            p.set_intermediate_callback(intermediate_callback)
            p.solve(x0)
            intermediate_callback.assert_called()
        intermediate_callback.reset()
        with self.subTest("Callback via constructor"):
            p = problem(n, intermediate_callback=intermediate_callback)
            p.solve(x0)
            intermediate_callback.assert_called()
        with self.subTest("Callback not returning a bool"):
            intermediate_callback = mock.Mock()
            p = problem(n, intermediate_callback=intermediate_callback)
            with self.assertRaises(RuntimeError):
                p.solve(x0)
