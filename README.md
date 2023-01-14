[![pipeline status](https://gitlab.com/ipyopt-devs/ipyopt/badges/main/pipeline.svg)](https://gitlab.com/ipyopt-devs/ipyopt/-/commits/main)
[![python version](https://img.shields.io/pypi/pyversions/ipyopt.svg?logo=python&logoColor=white)](https://pypi.org/project/ipyopt)
[![latest version](https://img.shields.io/pypi/v/ipyopt.svg)](https://pypi.org/project/ipyopt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Checked with pylint](https://img.shields.io/badge/pylint-checked-blue)](https://github.com/PyCQA/pylint)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3249818.svg)](https://doi.org/10.5281/zenodo.3249818)

# ipyopt

`ipyopt` is a Python 🐍 C++ extension that allows you to use
[Ipopt](http://www.coin-or.org/Ipopt/) in Python.

Ipopt solves general nonlinear programming problems of the form

```
min f(x)
```

under the constraints

```
g_l <= g(x) <= g_u,
x_l <= x <= x_u,
```

where `x` is `n` dimensional and `g(x)` is `m` dimensional.

## Goal

Provide as much performance as possible. This is also reflected in the
fact that the shipped `scipy.optimize.minimize` `ipopt` method
deviates in some concerns from the usual methods in scipy.
If you are interested in a more `scipy` like interface, have a look at [cyipopt](https://github.com/mechmotum/cyipopt).

## Installation

*Note* the pypi repository currently only provides 🐧 linux wheels.

```bash
pip install [--user] ipyopt
```

This will install a precompiled binary version from pypi. Please note,
that the precompiled binary is linked against the unoptimized
reference implementation of blas/lapack. If you want to take advantage
of optimized versions of blas/lapack, compile from source:

```bash
pip install --no-binary ipyopt ipyopt
```
In this case, you also need [Ipopt](https://github.com/coin-or/Ipopt) and
[Numpy](https://numpy.org/).
On a debian based system:

```bash
sudo apt-get install python3-numpy coinor-ipopt
```

If `coinor-ipopt` does not link correctly, you might have to compile
`ipopt` yourself.
See the section [Build](#build) below or [.ci/Dockerfile](.ci/Dockerfile) on
how this can be done.

## Usage

You can use `ipyopt` like this:

```python
import ipyopt
# define your call back functions
nlp = ipyopt.Problem(...)
nlp.solve(...)
```

For an example, see [examples/hs071.py](examples/hs071.py).

Note that the `ipyopt.Problem.solve(.)` mutates some its arguments, including the initial guess for the variables, and the multipliers.

For maximal performance, there is also support for [PyCapsules](https://docs.python.org/3/c-api/capsule.html) /
[scipy.LowLevelCallable](https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html). By
using this approach, there will be no C++ <-> Python interactions
during Ipopt optimization. Here is an example
[test/c_capsules/](test/c_capsules) (C code) and
[test/test_ipyopt.py](test/test_ipyopt.py) (Python code using the
PyCapsules provided by the C code).

For more details and hints, see the [docs](https://ipyopt-devs.gitlab.io/ipyopt).

## Build

`ipyopt` depends on the following packages:

1. A compiler and a linker, e.g. gcc, ld
2. [Ipopt](https://github.com/coin-or/Ipopt)
3. [Numpy](http://numpy.org/)
4. Python.h (part of the python source code, you can download it from
   [Python.org](https://python.org))

To build from source, first, get the latest source code using:

```sh
git clone https://gitlab.com/ipyopt-devs/ipyopt.git
```

Check whether a file `ipopt.pc` was distributed with your Ipopt installation.
If this is the case and `ipopt.pc` is in the search path of `pkg-config`
(on unix systems:
`/usr/lib/pkgconfig`, `/usr/share/pkgconfig`, `/usr/local/lib/pkgconfig`,
`/usr/local/share/pkgconfig`), nothing has to be modified.

In this case run

```sh
python setup.py build
sudo python setup.py install
```
	
If `pkg-config` is not available for your system, you will need to
pass appropriate information to `setup.py` by setting the environment
variable `CFLAGS`. Example:
```sh
CFLAGS="-I/usr/include/coin/ -l/usr/lib64 -lipopt -lmumps_common -ldmumps -lzmumps -lsmumps -lcmumps -llapack -lblas -lblas -lblas -lm  -ldl' ./setup.py build
sudo python setup.py install
```
	
If you have an `ipopt.pc` which is not in the `pkg-config` search path,
specify the path via the `PKG_CONFIG_PATH` environment variable (see below).
If you cannot find an `ipopt.pc` in your `ipopt` installation, there is an
example pc file [pkgconfig/ipopt.pc](pkgconfig/ipopt.pc).
Copy it to a location (best of all directly in a subfolder named
`pkgconfig` of your Ipopt installation) and edit it to reflect the
library and include paths of the dependencies.

Then do

```sh
PKG_CONFIG_PATH=<dir containing ipopt.pc> python setup.py build
sudo python setup.py install
```

## Testing

**Unit tests:**

```sh
python -m unittest
```

**Run examples:**

Use the following command under the
[examples](examples) directory. 

```sh
python hs071.py
```
	
The file [examples/hs071.py](examples/hs071.py) contains a toy
optimization problem. If everything is OK, `ipyopt` will invoke
`Ipopt` to solve it for you. This python file is self-documented and
can be used as a template for writing your own optimization problems.

**Hessian Estimation**: since Hessian estimation is usually tedious,
Ipopt can solve problems without Hessian estimation. `ipyopt` also
supports this feature. The file [examples/hs071.py](examples/hs071.py)
demonstrates the idea. If you provide the `ipyopt.Problem` constructor
with an `eval_h` callback function, `Ipopt` will delegate the Hessian matrix calculation to your
function (otherwise `Ipopt` will approximate Hessian for you).

## Contributing

1. Fork it.
2. Create a branch (`git checkout -b new_branch`)
3. Commit your changes (`git commit -am "your awesome message"`)
4. Push to the branch (`git push origin new_branch`)
5. Create a merge request

## Credits
* Modifications on logger made by OpenMDAO at NASA Glenn Research Center, 2010 and 2011
* Added "eval_intermediate_callback" by OpenMDAO at NASA Glenn Research Center, 2010 and 2011
* Modifications on the SAFE_FREE macro made by Guillaume Jacquenot, 2012
* Changed logger from code contributed by alanfalloon
* Originally developed by Eric Xu when he was a PhD student at
[Washington University](https://wustl.edu/) and issued under the BSD
license. Original repository: [xuy/pyipopt](https://github.com/xuy/pyipopt).
