#ifndef PY_IPOPT_LOGGING_
#define PY_IPOPT_LOGGING_

#include "Python.h"

#define LOG_INFO 20
#define LOG_OFF 0
#define LOG_FULL 10

void logger(int level, const char *fmt, ...);
void logger_register_log_levels(PyObject *module);
_Bool logger_set_loglevel(int level);

#endif				//  PY_IPOPT_LOGGING_
