[![DOI](https://zenodo.org/badge/143015117.svg)](https://zenodo.org/badge/latestdoi/143015117)

# IPyOpt

IPyOpt is a python module that allows you to use
[Ipopt](http://www.coin-or.org/Ipopt/) in Python.
It was developed by Eric Xu when he was a PhD student at [Washington
University](https://wustl.edu/) and issued under the BSD license.
Original repository: [xuy/pyipopt](https://github.com/xuy/pyipopt).

## Installation

### Dependencies

IPyOpt depends on the following packages:

1. A compiler and a linker, e.g. gcc, ld
2. [Ipopt](https://projects.coin-or.org/Ipopt)
3. [Numpy](http://numpy.scipy.org/)
4. Python.h (part of the python source code, you can download it from
   [Python.org](http://python.org))

### Install

First, get the latest source code using:

```sh
$ git clone http://github.com/g-braeunlich/IPyOpt.git
```

Check whether a file `ipopt.pc` was distributed with your Ipopt installation.
If this is the case and `ipopt.pc` is in the search path of `pkg-config`
(on unix systems:
`/usr/lib/pkgconfig`, `/usr/share/pkgconfig`, `/usr/local/lib/pkgconfig`,
`/usr/local/share/pkgconfig`), nothing has to be modified.

In this case run

```sh
$ python setup.py build
$ sudo python setup.py install
```
	
If `pkg-config` is not available for your system, you will need to
pass appropriate information to `setup.py` by setting the environment
variable `CFLAGS`. Example:
```sh
$ CFLAGS="-I/usr/include/coin/ -l/usr/lib64 -lipopt -lmumps_common -ldmumps -lzmumps -lsmumps -lcmumps -llapack -lblas -lblas -lblas -lm  -ldl' ./setup.py build
$ sudo python setup.py install
```
	
If you have an `ipopt.pc` which is not in the `pkg-config` search path,
specify the path via the `PKG_CONFIG_PATH` environment variable (see below).
If you cannot find an `ipopt.pc` in your `ipopt` installation, there is an
example pc file in the directory `pkgconfig`.
Copy it to a location (best of all directly in a subfolder named
`pkgconfig` of your Ipopt installation) and edit it to reflect the
library and include paths of the dependencies.

Then do

```sh
$ PKG_CONFIG_PATH=<dir containing ipopt.pc> python setup.py build
$ sudo python setup.py install
```

## Usage

You can use IPyOpt like this:

```python
import ipyopt
# define your call back functions
nlp = ipyopt.Problem(...)
nlp.solve(...)
```

You can also check out `examples/hs071.py` to see how to use IPyOpt.

IPyOpt as a module comes with docstring. You can poke around 
it by using Python's `help()` command.

## Testing

I have included an example 

To see if you have IPyOpt ready, use the following command under the
`examples`'s directory. 

```sh
$ python hs071.py
```
	
The file `hs071.py` contains a toy optimization problem. If everything
is OK, IPyOpt will invoke Ipopt to solve it for you. This python file
is self-documented and can be used as a template for writing your own
optimization problems. 

IPyOpt is a legitimate Python module, you can inspect it by using
standard Python commands like `dir` or `help`. All functions in
IPyOpt are documented in details.

**Hessian Estimation**: since Hessian estimation is usually tedious,
Ipopt can solve problems without Hessian estimation. IPyOpt also
supports this feature. The file `hs071.py` demonstrates the idea. If
you provide the `ipyopt.Problem` constructor with an `eval_h` callback
function as well as the `apply_new` callback function, Ipopt will
delegate the Hessian matrix calculation to your function (otherwise
Ipopt will approximate Hessian for you).

## Contributing

1. Fork it.
2. Create a branch (`git checkout -b new_branch`)
3. Commit your changes (`git commit -am "your awesome message"`)
4. Push to the branch (`git push origin new_branch`)
5. Create a pull request
6. Nag me about it if I am lazy.

## Troubleshooting

### Check Ipopt

IPyOpt links to Ipopt's C library. If that library is not available IPyOpt will fail
during module initialization. To check the availability of this library, you can go to
`$IPOPT_DIR/Ipopt/examples/hs071_c/`
and issue `make` to ensure you can compile and run the toy example supplied by Ipopt. 

### Miscellaneous problems

* Error:
  ```python
  import ipyopt
  ```
  ```
  ImportError: can not find  libipopt.so.0
  ```

* Solution:
  find it and copy it to a folder that ld can access

* Error:
  ```python
  import ipyopt
  ```
  ```
  ImportError: /usr/lib/libipopt.so.0: undefined symbol: _gfortran_XXX
  ```

* Solution: 
  check if your `hs071_c` example work. It is very likely that your
  ipopt library is not correctly compiled.

* Error:
  ```python
  import ipyopt
  ```
  ```
  ImportError: /usr/lib/libipopt.so.0: undefined symbol: SetIntermediateCallback
  ```

* Solution:
  SetIntermediateCallback is a function added since Ipopt 3.9.1.
  (see https://projects.coin-or.org/Ipopt/changeset/1830 )
  Make sure you have an Ipopt version >= 3.9.1

* Error:
  ```python
  import ipyopt
  ```
  ```
  ImportError: /usr/lib/libipopt.so.0: undefined symbol: ma19ad_
  ```

* Solution:
  First, use 
  ```sh
  nm /usr/lib/libipopt.so.0 | grep ma19ad_ 
  ```
  to see if it is marked with U. It should. This means that
  `libipopt.so.0` is not aware of `libcoinhsl.so.0`. You can fix this by
  adding `-lcoinhsl` to the `CFLAGS` variable (see section install). It seems to me that
  this happens in the recent versions of ipopt. Eventually IPyOpt
  will have a better building mechanism, and I will fix this soon. 

* Error:
  ```python
  import ipyopt
  ```
  ```
  ImportError: /usr/lib/libipopt.so.0: undefined symbol: SomeKindOfSymbol
  ```
	
* Solution:
  I can assure you that it is NOT a bug of IPyOpt. It is very
  likely that you did not link the right package when compiling
  IPyOpt. 
	
  First, use 
  ```sh
  nm /usr/lib/libipopt.so.0 | grep SomeKindOfSymbol
  ```
  to see if this symbol is indeed missing. Do a Google search to find the library file, and 
  add `-lWhateverLibrary` to the `CFLAGS` variable (see section install). 
	
  Ipopt is built using various third-party libraries. Different
  machines may have different set of libraries. You should 
  try to locate these dependencies and indicate them when compiling
  IPyOpt. This is just a limitation of dynamic linking libraries
  and is not related to IPyOpt. Please do not report a missing symbol
  error as a "bug" to me unless you are 100% sure it is the problem
  of IPyOpt.
	

## Contact

Gerhard Bräunlich <g.braeunlich@disroot.org>

## Credits
* Modifications on logger made by OpenMDAO at NASA Glenn Research Center, 2010 and 2011
* Added "eval_intermediate_callback" by OpenMDAO at NASA Glenn Research Center, 2010 and 2011
* Modifications on the SAFE_FREE macro made by Guillaume Jacquenot, 2012
* Changed logger from code contributed by alanfalloon
