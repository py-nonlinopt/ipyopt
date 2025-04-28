#!/bin/env python3
"""Backport pyproject.toml to setup.cfg.

This is necessary for python 3.6 only.
"""

import tomllib

with open("pyproject.toml", "rb") as stream:
    metadata = tomllib.load(stream)
project = metadata["project"]
author, author_email = zip(
    *((author["name"], author["email"]) for author in project["authors"])
)
packages = metadata["tools"]["setuptools"]["packages"]

print(  # noqa: T201
    f"""[metadata]
name = {project["name"]}
author = {", ".join(author)}
author_email = {", ".join(author_email)}
description = {project["description"]}
long_description = file: README.md
classifiers =
    {"\n    ".join(project["classifiers"])}

[options]
install_requires =
    {"\n    ".join(packages)}
zip_safe = False
include_package_data = True

[options.package_data]
{project["name"]} = py.typed
"""
)
