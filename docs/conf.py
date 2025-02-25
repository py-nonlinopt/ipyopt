"""Configuration file for the Sphinx documentation builder."""  # noqa: INP001

import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = "ipyopt"
copyright = "2021-2023, ipyopt developers"  # noqa: A001
author = "Gerhard Bräunlich, Nikitas Rontsis"


extensions = ["sphinx.ext.napoleon", "sphinx_rtd_theme", "sphinx.ext.mathjax"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
autodoc_mock_imports = ["ipyopt.ipyopt"]
