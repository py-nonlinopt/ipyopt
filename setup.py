#!/usr/bin/env python

"""
Originally contributed by Lorne McIntosh.
Modified by Eric Xu
Further modification by random internet people.
"""

import warnings
try:
    from setuptools import setup, Extension
except ImportError:
    raise RuntimeError('setuptools is required')

from setup_helpers import LazyList


def load_extensions():
    """Gets plugged into a LazyList to prevent numpy import until numpy
    is installed"""
    yield Extension("ipyopt",
                    sources=["src/callback.c",
                             "src/ipyopt_module.c", "src/logger.c"],
                    **get_compiler_flags())


def get_compiler_flags():
    """Tries to find all needed compiler flags needed to compile the extension
    """
    import os
    from numpy import get_include as _numpy_get_include
    compiler_flags = {"include_dirs": [_numpy_get_include()]}
    try:
        from setup_helpers import pkg_config
        return pkg_config("ipopt", **compiler_flags)
    except (RuntimeError, FileNotFoundError) as e:
        if 'CFLAGS' not in os.environ:
            warnings.warn(
                "pkg-config not installed or malformed pc file.\n"
                "Message from pkg-config:\n{}\n\n"
                "You have to provide setup.py with the include and library "
                "directories of IPOpt. Example:\n"
                "CFLAGS='-I/usr/include/coin/ -l/usr/lib64 "
                "-lipopt -lmumps_common -ldmumps -lzmumps -lsmumps "
                "-lcmumps -llapack -lblas -lblas -lblas "
                "-lm  -ldl' ./setup.py build".format(e.args[0]))
        return compiler_flags


url = "https://github.com/g-braeunlich/ipyopt"

setup(
    name="ipyopt",
    version="0.9.2",
    description="An IPOpt connector for Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gerhard Bräunlich",
    author_email="g.braeunlich@disroot.org",
    url=url,
    ext_modules=LazyList(load_extensions()),
    install_requires=["numpy"],
)
