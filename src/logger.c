#include <stdio.h>
#include "logger.h"

typedef struct { int level; const char *name; } named_log_level;
static const named_log_level named_log_levels[] = {
  {
    .level=LOG_OFF,
    .name="LOGGING_OFF",
  },
  {
    .level=LOG_INFO,
    .name="LOGGING_INFO",
  },
  {
    .level=LOG_FULL,
    .name="LOGGING_DEBUG",
  }
 };

#define N_named_log_levels (sizeof(named_log_levels) / sizeof(named_log_level))

static int user_log_level = LOG_INFO;

_Bool logger_set_loglevel(int level)
{
  if(level < 0)
    {
      PyErr_Format(PyExc_ValueError, "Negative log levels are not allowed");
      return 0;
    }
  user_log_level = level;
  return 1;
}

void logger(int level, const char *fmt, ...)
{
  if(level >= user_log_level && user_log_level != LOG_OFF)
    {
      va_list ap;
      va_start(ap, fmt);
      PySys_WriteStdout("[IPyOpt] ");
      PySys_WriteStdout(fmt, ap);
      va_end(ap);
      PySys_WriteStdout("\n");
    }
}

void logger_register_log_levels(PyObject *module)
{
  unsigned long i;
  for(i=0; i<N_named_log_levels; i++)
    PyModule_AddIntConstant(module, named_log_levels[i].name, named_log_levels[i].level);
}
