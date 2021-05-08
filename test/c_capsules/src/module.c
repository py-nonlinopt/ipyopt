#include "Python.h"
#include <stdbool.h>

/** PyCapsule implementation of the NLP defined in test_ipyopt.py
 *
 * h and intermediate callback also can be given a python callable as context.
 * In this case, this capsules will call the callable if called itself.
 * This is for testing purposes (see test_ipyopt.py)
 * To get / set the context of a capsule, see the doc of
 * capsule_get_context / capsule_set_context below, or via
 * help(c_capsules.capsule_set_context) in python.
 */

#define N 4
#define M 1

static bool f(int n, const double *x, double *obj_value, void *userdata) {
  double S = 0;
  unsigned int i;
  for (i = 0; i < N; i++)
    S += x[i] * x[i];
  *obj_value = S;
  return true;
}

static bool grad_f(int n, const double *x, double *out, void *userdata) {
  unsigned int i;
  for (i = 0; i < N; i++)
    out[i] = 2.0 * x[i];
  return true;
}

const static double e_x[] = {1., 0., 0., 0.};

static bool g(int n, const double *x, int m, double *out, void *userdata) {
  double S = 0, dx;
  unsigned int i;
  for (i = 0; i < N; i++) {
    dx = x[i] - e_x[i];
    S += dx * dx;
  }
  out[0] = S;
  return true;
}

static bool jac_g(int n, const double *x, int m, int n_out, double *out,
                  void *userdata) {
  int i;
  for (i = 0; i < n_out; i++)
    out[i] = 2.0 * (x[i] - e_x[i]);
  return true;
}

static void call_callback(PyObject *obj) {
  if (obj != NULL && PyCallable_Check(obj))
    PyObject_Call(obj, PyTuple_New(0), NULL);
}

static bool h(int n, const double *x, double obj_factor, int m,
              const double *lambda, int n_out, double *out, void *userdata) {
  int i;
  call_callback(userdata);
  for (i = 0; i < n_out; i++)
    out[i] = 2.0 * (obj_factor + lambda[0]);
  return true;
}

static bool intermediate_callback(int algorithm_mode, int iter,
                                  double obj_value, double inf_pr,
                                  double inf_du, double mu, double d_norm,
                                  double regularization_size, double alpha_du,
                                  double alpha_pr, int ls_trails,
                                  const void *ip_data, void *ip_cq,
                                  void *userdata) {
  call_callback(userdata);
  return true;
}

static bool check_py_capsule(PyObject *obj, const char *fn_name) {
  if (!PyCapsule_CheckExact(obj)) {
    PyErr_Format(PyExc_ValueError,
                 "c_capsules.%s() arument 1 must be PyCapsule, not %s", fn_name,
                 Py_TYPE(obj)->tp_name);
    return false;
  }
  const char *name = PyCapsule_GetName(obj);

  if (!PyCapsule_IsValid(obj, name)) {
    PyErr_Format(PyExc_ValueError,
                 "c_capsules.%s() arument 1: Invalid PyCapsule with name '%s'",
                 fn_name, (name != NULL) ? name : "");
    return false;
  }
  return true;
}

static char DOC_CAPSULE_SET_CONTEXT[] =
    "capsule_set_context(capsule: PyCapsule, obj: Any)\n"
    "Take a python object obj and set its"
    " pointer at C level as the context to the capsule.\n"
    "If obj is None, set the context to NULL.\n"
    "Use at own risk! It can be usefull to check whether"
    " a low level callable has been called.";
static PyObject *py_capsule_set_context(PyObject *self, PyObject *args) {
  PyObject *capsule = NULL;
  PyObject *context = NULL;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &context) ||
      !check_py_capsule(capsule, "capsule_set_context"))
    return NULL;
  if (context == Py_None)
    context = NULL;
  PyCapsule_SetContext(capsule, context);
  Py_RETURN_NONE;
}

static char DOC_CAPSULE_GET_CONTEXT[] =
    "capsule_get_context(capsule: PyCapsule) -> Any\n"
    "Fetch the context of a capsule. If"
    " the context is NULL, returns None. Otherwise the C pointer"
    " is interpreted as a Python object pointer and the"
    " corresponding object is returned.\n"
    "This can result in segmentation faults, if the context is not"
    " pointing to a Python object at C level. Use at own risk!";
static PyObject *py_capsule_get_context(PyObject *self, PyObject *args) {
  PyObject *capsule = NULL;
  if (!PyArg_ParseTuple(args, "O", &capsule) ||
      !check_py_capsule(capsule, "capsule_set_context"))
    return NULL;
  PyObject *context = PyCapsule_GetContext(capsule);
  if (PyErr_Occurred())
    return NULL;
  if (context == NULL)
    Py_RETURN_NONE;
  Py_XINCREF(context);
  return context;
}

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "c_capsules",
    .m_doc = "c pycapsule",
    .m_size = -1,
    .m_methods = (PyMethodDef[]){{"capsule_set_context", py_capsule_set_context,
                                  METH_VARARGS, DOC_CAPSULE_SET_CONTEXT},
                                 {"capsule_get_context", py_capsule_get_context,
                                  METH_VARARGS, DOC_CAPSULE_GET_CONTEXT},
                                 {NULL, NULL, 0, NULL}},
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL,
};

PyMODINIT_FUNC PyInit_c_capsules(void) {
  PyObject *module = PyModule_Create(&moduledef);
  ;
  if (module == NULL)
    return NULL;

  PyObject *py_f = PyCapsule_New((void *)f, "c_capsules.f", NULL);
  PyObject *py_grad_f =
      PyCapsule_New((void *)grad_f, "c_capsules.grad_f", NULL);
  PyObject *py_g = PyCapsule_New((void *)g, "c_capsules.g", NULL);
  PyObject *py_jac_g = PyCapsule_New((void *)jac_g, "c_capsules.jac_g", NULL);
  PyObject *py_h = PyCapsule_New((void *)h, "c_capsules.h", NULL);
  PyObject *py_intermediate_callback = PyCapsule_New(
      (void *)intermediate_callback, "c_capsules.intermediate_callback", NULL);
  if (PyModule_AddObject(module, "f", py_f) < 0 ||
      PyModule_AddObject(module, "grad_f", py_grad_f) < 0 ||
      PyModule_AddObject(module, "g", py_g) < 0 ||
      PyModule_AddObject(module, "jac_g", py_jac_g) < 0 ||
      PyModule_AddObject(module, "h", py_h) < 0 ||
      PyModule_AddObject(module, "intermediate_callback",
                         py_intermediate_callback) < 0 ||
      PyModule_AddIntConstant(module, "n", N) < 0 ||
      PyModule_AddIntConstant(module, "m", M) < 0) {
    Py_XDECREF(py_f);
    Py_XDECREF(py_grad_f);
    Py_XDECREF(py_g);
    Py_XDECREF(py_jac_g);
    Py_XDECREF(py_h);
    Py_XDECREF(py_intermediate_callback);
    Py_DECREF(module);
    return NULL;
  }

  return module;
}
