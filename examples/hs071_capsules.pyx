import cython
cdef extern from "stdbool.h":
    ctypedef bint bool 

cdef api:
    bool f(int n, const double *x, double *obj_value, void* userdata):
        obj_value[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
        return True

    bool grad_f(int n, const double *x, double *out, void *userdata):
        out[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2])
        out[1] = x[0] * x[3]
        out[2] = x[0] * x[3] + 1.0
        out[3] = x[0] * (x[0] + x[1] + x[2])
        return True

    bool g(int n, const double *x, int m, double *out, void *userdata):
        out[0] = x[0] * x[1] * x[2] * x[3]
        out[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
        return True

    bool jac_g(int n, const double *x, int m, int n_out, double *out,
                      void *userdata):
        out[0] = x[1] * x[2] * x[3]
        out[1] = x[0] * x[2] * x[3]
        out[2] = x[0] * x[1] * x[3]
        out[3] = x[0] * x[1] * x[2]
        out[4] = 2.0 * x[0]
        out[5] = 2.0 * x[1]
        out[6] = 2.0 * x[2]
        out[7] = 2.0 * x[3]
        out[8] = x[0] * x[1] * x[2] * x[3]
        return True

    bool h(int n, const double *x, double obj_factor, int m,
                  const double *lagrange, int n_out, double *out, void *userdata):
        out[0] = obj_factor * (2 * x[3])
        out[1] = obj_factor * (x[3])
        out[2] = 0
        out[3] = obj_factor * (x[3])
        out[4] = 0
        out[5] = 0
        out[6] = obj_factor * (2 * x[0] + x[1] + x[2])
        out[7] = obj_factor * (x[0])
        out[8] = obj_factor * (x[0])
        out[9] = 0
        out[1] += lagrange[0] * (x[2] * x[3])

        out[3] += lagrange[0] * (x[1] * x[3])
        out[4] += lagrange[0] * (x[0] * x[3])

        out[6] += lagrange[0] * (x[1] * x[2])
        out[7] += lagrange[0] * (x[0] * x[2])
        out[8] += lagrange[0] * (x[0] * x[1])
        out[0] += lagrange[1] * 2
        out[2] += lagrange[1] * 2
        out[5] += lagrange[1] * 2
        out[9] += lagrange[1] * 2
        return True
