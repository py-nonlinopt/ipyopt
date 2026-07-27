#!/usr/bin/env python3
"""Legacy setup.py to build the C extensions."""

import os
import subprocess
import sys
import warnings
from datetime import datetime

from numpy import get_include as _numpy_get_include
from setuptools import Extension, setup

# 0.0.0-dev.* version identifiers for development only
__version__ = "0.0.0dev" + datetime.now().strftime("%Y%m%d")


def main():
    """Entry point."""
    setup(
        version=__version__,
        packages=["ipyopt"],
        ext_modules=[
            Extension(
                "ipyopt.ipyopt",
                sources=[
                    "src/ipyopt_module.cpp",
                    "src/py_nlp.cpp",
                    "src/nlp_base.cpp",
                ],
                depends=[
                    "src/c_nlp.hpp",
                    "src/nlp_base.hpp",
                    "src/nlp_builder.hpp",
                    "src/py_helpers.hpp",
                    "src/py_nlp.hpp",
                ],
                language="c++",
                **get_compiler_flags(),
            )
        ],
    )


def get_compiler_flags():
    """Tries to find all needed compiler flags needed to compile the extension."""
    compiler_flags = {"include_dirs": [_numpy_get_include()]}

    # On windows, Python extensions usually compile with MSVC (where we
    # usually don't use pkg-config). Builds against a MinGW compiled Ipopt
    # (e.g. the wheel builds, see .github/workflows/wheels.yml) opt in to
    # pkg-config by setting IPYOPT_USE_PKG_CONFIG=1; the extension must then
    # also be compiled with MinGW (`[build_ext] compiler = mingw32`), since
    # Ipopt's C++ interface requires a single common C++ toolchain.
    use_pkg_config = sys.platform != "win32" or os.environ.get(
        "IPYOPT_USE_PKG_CONFIG", ""
    ).lower() in {"1", "true", "yes", "on"}

    if not use_pkg_config:
        compiler_flags["extra_compile_args"] = ["/std:c++17"]
        try:
            return msvc_config(**compiler_flags)
        except (RuntimeError, FileNotFoundError) as e:
            warnings.warn(
                "No compatible MSVC configuration / directory found.\n"
                f"Message from MSVC configuration:\n{e.args[0]}\n\n"
                "This extension assumes the official release binaries (dlls) "
                "in the directory specified by the environment variable "
                "IPOPT_DIR"
                "You have to provide setup.py with the include and library "
                "directories of Ipopt. Example via environment:\n"
                "IPOPT_DIR='C:/path/to/Ipopt' ./setup.py build",
                stacklevel=0,
            )
            return compiler_flags

    # For all other cases, we try pkg_config
    compiler_flags["extra_compile_args"] = ["-std=c++17", "-O2"]
    try:
        return pkg_config("ipopt", **compiler_flags)
    except (RuntimeError, FileNotFoundError) as e:
        if "CFLAGS" not in os.environ:
            warnings.warn(
                "pkg-config not installed or malformed pc file.\n"
                f"Message from pkg-config:\n{e.args[0]}\n\n"
                "You have to provide setup.py with the include and library "
                "directories of Ipopt. Example:\n"
                "CFLAGS='-I/usr/include/coin/ -l/usr/lib64 "
                "-lipopt -lmumps_common -ldmumps -lzmumps -lsmumps "
                "-lcmumps -llapack -lblas -lblas -lblas "
                "-lm  -ldl' ./setup.py build",
                stacklevel=0,
            )
        return compiler_flags


def pkg_config(*packages, **kwargs):
    """Call pkg-config.

    Return all arguments for Extension() needed to compile the extension as a dict
    """
    flag_map = {
        b"-I": "include_dirs",
        b"-L": "library_dirs",
        b"-l": "libraries",
        b"-D": "define_macros",
    }
    try:
        res = subprocess.run(  # noqa: S603, UP022
            ("pkg-config", "--libs", "--cflags", *packages),  # noqa: S607
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode()) from e
    for token in res.stdout.split():
        flag = flag_map.get(token[:2])
        kwargs.setdefault(flag, []).append(token[2:].decode())
    define_macros = kwargs.get("define_macros")
    if define_macros:
        kwargs["define_macros"] = [
            # Set None as the value of the define if no value is passed:
            tuple([*d.split(), None][:2])
            for d in define_macros
        ]
    undefined_flags = kwargs.pop(None, None)
    if undefined_flags:
        warnings.warn(
            f"Ignoring flags {', '.join(undefined_flags)} from pkg-config", stacklevel=0
        )
    return kwargs


def msvc_config(**kwargs):
    """Returns the additional extension arguments for MSVC.

    For windows, we need to compile with MSVC and then usally don't work
    with pkg_config. We try to read an environment variable containing the
    directory or assume a subdirectory of the current directory.
    """
    ipopt_dir = os.environ.get("IPOPT_DIR", "./Ipopt")
    if not os.path.exists(ipopt_dir):
        errmsg = (
            "IPOPT_DIR environment variable not set to a valid path "
            "and no local Ipopt directory found."
        )
        raise FileNotFoundError(errmsg)

    win_dll_dir = os.path.join(ipopt_dir, "bin")

    win_dll_files = [
        os.path.join(win_dll_dir, dll)
        for dll in os.listdir(win_dll_dir)
        if dll.endswith(".dll")
    ]

    # Check the corresponding lib files exist
    win_lib_dir = os.path.join(ipopt_dir, "lib")
    win_lib_files = [
        os.path.splitext(lib)[0]
        for lib in os.listdir(win_lib_dir)
        if lib.endswith(".lib")
    ]

    include_dir = os.path.join(ipopt_dir, "include", "coin-or")

    compiler_flags = {
        "include_dirs": [include_dir],
        "library_dirs": [win_lib_dir],
        "libraries": win_lib_files,
        "data_files": [("", win_dll_files)],
    }

    for flag, value in kwargs.items():
        existing_flags = compiler_flags.setdefault(flag, [])
        existing_flags += value

    return compiler_flags


main()
