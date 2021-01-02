#ifndef _PY_HELPERS_H_
#define _PY_HELPERS_H_

#include "Python.h"

template <typename... Args> PyObject *py_tuple(Args... objects) {
  constexpr std::size_t n = sizeof...(Args);
  return PyTuple_Pack(n, objects...);
}

template <typename... Args>
static PyObject *py_call(PyObject *callback, Args... args) {
  if (callback == nullptr) {
    PyErr_Format(PyExc_RuntimeError, "python callback is nullptr");
    return nullptr;
  }
  auto tuple = py_tuple(args...);
  if (tuple == nullptr) {
    PyErr_Format(PyExc_MemoryError,
                 "Could not pack python arguments for python callable");
    return nullptr;
  }
  PyObject *result = PyObject_Call(callback, tuple, nullptr);
  Py_CLEAR(tuple);
  return result;
}

#endif
