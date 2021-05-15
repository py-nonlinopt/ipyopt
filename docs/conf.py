# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = "ipyopt"
copyright = "2021, Gerhard Bräunlich"
author = "Gerhard Bräunlich"


extensions = ["sphinx.ext.napoleon", "sphinx_rtd_theme", "sphinx.ext.mathjax"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
