#ifndef _PY_HELPERS_H_
#define _PY_HELPERS_H_

#include "Python.h"
#include <array>
#include <optional>
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

/**
 * @brief Converts a c string to a str PyObject.
 */
inline PyObject *to_py_object(const char *val) {
  return PyUnicode_FromString(val);
}
/**
 * @brief Converts an int to a int PyObject.
 */
inline PyObject *to_py_object(const int val) { return PyLong_FromLong(val); }
/**
 * @brief "Converts" a PyObject to PyObject (This allow passing PyObject to `py_dict`).
 */
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

/**
 * @brief Converts a PyTuple to a std::array<PyObject*, N>.
 *
 * Returns an array filled with nullptr and sets the python error indicator on failure.
 */
template <std::size_t N>
std::array<PyObject *, N> from_py_tuple(PyObject *obj,
                                        const char *err_context) {
  if (!PyTuple_Check(obj)) {
    PyErr_Format(PyExc_TypeError, "%s: a tuple is needed.", err_context);
    return std::array<PyObject *, N>{};
  }
  if (PyTuple_Size(obj) != N) {
    PyErr_Format(PyExc_TypeError,
                 "%s: a tuple of size %d is needed. Found tuple of size %d",
                 err_context, N, PyTuple_Size(obj));
    return std::array<PyObject *, N>{};
  }
  auto t = std::array<PyObject *, N>{};
  for (std::size_t i = 0; i < N; i++)
    t[i] = PyTuple_GetItem(obj, i);
  return t;
}

template <typename T> std::optional<T> from_py(PyObject *obj);

/**
 * @brief Converts a PyObject int to a C int.
 *
 * Returns std::nullopt on failure.
 */
template <> inline std::optional<int> from_py(PyObject *obj) {
  auto ret = PyLong_AsLong(obj);
  if (PyErr_Occurred())
    return std::nullopt;
  return ret;
}

/**
 * @brief Converts a PySequence to a std::vector<T>, parsing items using from_py.
 *
 * Returns an empty vector and sets the python error indicator on failure.
 */
template <typename T>
std::vector<T> from_py_sequence(PyObject *obj, const char *err_context) {
  auto sequence = PySequence_Fast(obj, "");
  if (sequence == nullptr) {
    PyErr_Format(PyExc_TypeError, "%s: a sequence is needed.", err_context);
    return std::vector<T>{};
  }
  const auto size = PySequence_Fast_GET_SIZE(sequence);
  if (size < 0) {
    PyErr_Format(PyExc_RuntimeError, "%s: Got negative size", err_context);
    return std::vector<T>{};
  }
  auto vec = std::vector<T>(size);
  auto **seq_items = PySequence_Fast_ITEMS(sequence);
  for (std::size_t i = 0; i < (std::size_t)size; i++) {
    if (auto val = from_py<T>(seq_items[i]))
      vec[i] = val.value();
    else {
      Py_XDECREF(sequence);
      PyErr_Format(PyExc_TypeError, "%s[%d]: invalid type. Expected int",
                   err_context, i);
      return std::vector<T>{};
    }
  }
  Py_XDECREF(sequence);

  return vec;
}

#endif
