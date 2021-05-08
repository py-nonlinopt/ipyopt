#!/usr/bin/env python3

"""setup the c_capsules test module for the unit tests.

Not needed for the actual module, but used in CI.
"""

from setuptools import setup, Extension

extensions = [Extension("test.c_capsules", sources=["test/c_capsules/src/module.c"])]

setup(
    name="ipyopt.test",
    package_dir={"test.c_capsules": "test"},
    zip_safe=False,
    ext_modules=extensions,
    install_requires=["numpy", "scipy"],
)
