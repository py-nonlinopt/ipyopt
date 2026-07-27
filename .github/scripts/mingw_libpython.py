"""Make the CPython import library visible to MinGW's linker.

python.org CPython ships only ``libs/python3XY.lib`` (an MSVC-style import
library). GNU ld understands that file format, but when given ``-lpython3XY``
it searches for ``libpython3XY.dll.a`` / ``libpython3XY.a`` style names (only
recent binutils also try ``python3XY.lib``). Copying the import library to
``libpython3XY.a`` makes the lookup work with any binutils.

Run by cibuildwheel as a `before-build` step on Windows (see pyproject.toml),
once per Python version, with the build environment's Python.
"""

import shutil
import sys
from pathlib import Path

libs = Path(sys.base_prefix) / "libs"
name = f"python{sys.version_info[0]}{sys.version_info[1]}"
src = libs / f"{name}.lib"
dst = libs / f"lib{name}.a"
if src.exists() and not dst.exists():
    shutil.copy(src, dst)
    print(f"Copied {src} -> {dst}")
else:
    print(f"Nothing to do ({src} -> {dst})")
