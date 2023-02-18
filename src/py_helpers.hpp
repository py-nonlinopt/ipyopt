#ifndef _PY_HELPERS_H_
#define _PY_HELPERS_H_

#include "Python.h"
#include <tuple>

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

inline PyObject *to_py_object(const char *val) {
  return PyUnicode_FromString(val);
}
inline PyObject *to_py_object(const int val) { return PyLong_FromLong(val); }
inline PyObject *to_py_object(PyObject *obj) { return obj; }

namespace detail {
template <typename T>
void py_dict_add_key_val_pairs(PyObject *dict,
                               std::tuple<const char *, T> key_val_pair) {
  auto &[key, val] = key_val_pair;
  auto *obj = to_py_object(val);
  PyDict_SetItemString(dict, key, obj);
  Py_XDECREF(obj);
}

template <typename T, typename... Args>
void py_dict_add_key_val_pairs(PyObject *dict,
                               std::tuple<const char *, T> key_val_pair_0,
                               Args... key_val_pairs) {
  py_dict_add_key_val_pairs(dict, key_val_pair_0);
  py_dict_add_key_val_pairs(dict, key_val_pairs...);
}

} // namespace detail

/**
 * @brief Creates a new PyDict containing `key_val_pairs...`.
 *
 * `key_val_pairs` should be instances of std::tuple<const char*, T>.

 * If T=PyObject*, this functions "steals" references from `key_val_pairs...`
 * (the refcount of the objects will not be increased and will be decreased
 * once the dict is garbage collected).
 */
template <typename... Args> PyObject *py_dict(Args... key_val_pairs) {
  auto *dict = PyDict_New();
  detail::py_dict_add_key_val_pairs(dict, key_val_pairs...);
  return dict;
}

#endif
