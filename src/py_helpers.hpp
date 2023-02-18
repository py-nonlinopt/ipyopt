#ifndef _PY_HELPERS_H_
#define _PY_HELPERS_H_

#include "Python.h"

/**
 * @brief Creates a new PyTuple containing `objects...`.
 *
 * This functions "steals" references from `objects...`
 * (the refcount of the objects will not be increased and will be decreased
 * once the tuple is garbage collected).
 */
template <typename... Args> PyObject *py_tuple(Args... objects) {
  constexpr std::size_t n = sizeof...(Args);
  auto *t = PyTuple_New(n);
  auto pos = std::size_t{0};
  for (auto obj : {objects...})
    PyTuple_SET_ITEM(t, pos++, obj);
  return t;
}

/**
 * @brief Calls `callback`, with arguments `args...` and
 * returns the return value as a PyObject.
 *
 * This functions "steals" references from `args...`:
 * After the call to `callback` all arguments will have their ref counter decreased by 1.
 */
template <typename... Args>
PyObject *py_call(PyObject *callback, Args... args) {
  if (callback == nullptr) {
    PyErr_Format(PyExc_RuntimeError, "python callback is nullptr");
    return nullptr;
  }
  auto *tuple = py_tuple(args...);
  if (tuple == nullptr) {
    PyErr_Format(PyExc_MemoryError,
                 "Could not pack python arguments for python callable");
    return nullptr;
  }
  auto *result = PyObject_Call(callback, tuple, nullptr);
  Py_CLEAR(tuple);
  return result;
}

#endif
