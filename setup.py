#!/usr/bin/env python3

from datetime import datetime
import os
import warnings
import subprocess
from numpy import get_include as _numpy_get_include
from setuptools import setup, Extension

# 0.0.0-dev.* version identifiers for development only
__version__ = "0.0.0.dev" + datetime.now().strftime("%Y%m%d")


def main():
    setup(
        name="ipyopt",
        version=__version__,
        description="An IPOpt connector for Python",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author="Gerhard Bräunlich",
        author_email="g.braeunlich@disroot.org",
        url="https://github.com/g-braeunlich/ipyopt",
        ext_modules=[
            Extension(
                "ipyopt",
                sources=[
                    "src/ipyopt_module.cpp",
                    "src/py_nlp.cpp",
                    "src/nlp_base.cpp",
                ],
                depends=[
                    "src/nlp_base.hpp",
                    "src/py_helpers.hpp",
                    "src/py_nlp.hpp",
                ],
                language="c++",
                extra_compile_args=["-std=c++17"],
                **get_compiler_flags()
            )
        ],
        install_requires=["numpy"],
    )


def get_compiler_flags():
    """Tries to find all needed compiler flags needed to compile the extension"""
    compiler_flags = {"include_dirs": [_numpy_get_include()]}
    try:
        return pkg_config("ipopt", **compiler_flags)
    except (RuntimeError, FileNotFoundError) as e:
        if "CFLAGS" not in os.environ:
            warnings.warn(
                "pkg-config not installed or malformed pc file.\n"
                "Message from pkg-config:\n{}\n\n"
                "You have to provide setup.py with the include and library "
                "directories of IPOpt. Example:\n"
                "CFLAGS='-I/usr/include/coin/ -l/usr/lib64 "
                "-lipopt -lmumps_common -ldmumps -lzmumps -lsmumps "
                "-lcmumps -llapack -lblas -lblas -lblas "
                "-lm  -ldl' ./setup.py build".format(e.args[0])
            )
        return compiler_flags


def pkg_config(*packages, **kwargs):
    """Calls pkg-config returning a dict containing all arguments
    for Extension() needed to compile the extension
    """
    flag_map = {
        b"-I": "include_dirs",
        b"-L": "library_dirs",
        b"-l": "libraries",
        b"-D": "define_macros",
    }
    try:
        res = subprocess.run(
            ("pkg-config", "--libs", "--cflags") + packages,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode()) from e
    for token in res.stdout.split():
        kwargs.setdefault(flag_map.get(token[:2]), []).append(token[2:].decode())
    define_macros = kwargs.get("define_macros")
    if define_macros:
        kwargs["define_macros"] = [tuple(d.split()) for d in define_macros]
    undefined_flags = kwargs.pop(None, None)
    if undefined_flags:
        warnings.warn(
            "Ignoring flags {} from pkg-config".format(", ".join(undefined_flags))
        )
    return kwargs


main()
