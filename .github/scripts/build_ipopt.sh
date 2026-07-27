#!/usr/bin/env bash
# Build Ipopt and the libraries it depends on (BLAS/LAPACK + MUMPS) from source.
#
# Runs on all three platforms:
#   - Linux: as cibuildwheel's `before-all` step inside the manylinux
#     container (as root), prefix /usr/local.
#     Mirrors what .ci/Dockerfile does for the GitLab pipeline.
#   - macOS: as cibuildwheel's `before-all` step on the runner,
#     prefix $HOME/ipopt.
#   - Windows: from the workflow's MSYS2 (UCRT64) shell, prefix C:/ipopt.
#     Everything is compiled with MinGW gcc/gfortran -- the official coin-or
#     binary releases are built with Intel compilers + MKL and ship
#     libiomp5md.dll (Intel's OpenMP runtime), which is not safe to vendor
#     into a wheel: a second OpenMP runtime in the same process (e.g. from
#     numpy/MKL or torch) leads to the well-known "OMP Error #15" crashes.
#     The Python extension itself is compiled with MinGW too (setup.cfg
#     `compiler = mingw32`, written by the workflow): ipyopt uses Ipopt's
#     C++ interface, so extension and libipopt must share one C++ toolchain.
#
# MUMPS is built by coin-or's ThirdParty-Mumps in its default configuration:
# sequential (ships its own MPI stub, libseq) and without OpenMP. This is
# deliberate: a wheel that vendors an OpenMP runtime can clash with any other
# wheel (numpy/scipy/torch, ...) loading a different -- or a second copy of
# the same -- OpenMP runtime into the process. OpenMP-enabled variants would
# need to be separate wheels/packages, one per OpenMP runtime.

set -euxo pipefail

LAPACK_VERSION="${LAPACK_VERSION:-3.12.1}"
COIN_MUMPS_VERSION="${COIN_MUMPS_VERSION:-3.0.9}"
IPOPT_VERSION="${IPOPT_VERSION:-3.14.17}"

NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
workdir="$(mktemp -d)"
cd "$workdir"

case "$(uname)" in
  Linux)
    PREFIX=/usr/local
    # manylinux_2_28 ships gcc-toolset (incl. gfortran), cmake and pkg-config;
    # these are only fallbacks in case the image changes:
    command -v pkg-config >/dev/null || dnf install -y pkgconf-pkg-config || yum install -y pkgconfig
    command -v cmake >/dev/null || dnf install -y cmake || pipx install cmake
    command -v gfortran >/dev/null || dnf install -y gcc-gfortran || yum install -y gcc-gfortran

    # Reference BLAS/LAPACK, same as the GitLab CI image (.ci/Dockerfile).
    # auditwheel later vendors liblapack/libblas (+ libgfortran) into the wheel.
    curl -fsSL "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v${LAPACK_VERSION}.tar.gz" | tar xz
    cmake -S "lapack-${LAPACK_VERSION}" -B lapack-build \
      -DCBLAS=ON -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib
    cmake --build lapack-build -j"$NPROC" --target install

    LAPACK_LFLAGS="-L$PREFIX/lib -llapack -lblas"
    export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    ;;

  Darwin)
    PREFIX="$HOME/ipopt"
    # Fortran compiler for MUMPS (the Homebrew gcc formula ships an
    # unversioned `gfortran`):
    brew install gcc
    export FC=gfortran

    # BLAS/LAPACK: Apple's Accelerate framework -> nothing to vendor.
    LAPACK_LFLAGS="-framework Accelerate"

    # Statically link the GNU Fortran runtime into libcoinmumps:
    # Homebrew dylibs are built for the runner's macOS version, so vendoring
    # libgfortran.dylib would force the wheel's platform tag up to that
    # version (delocate refuses to tag lower). Instead, expose only the
    # static archives in a directory that ld searches first -- with no dylib
    # to prefer, ld links them statically. If delocate still reports a
    # Homebrew dylib in the wheel, add its archive to the list below, or
    # raise MACOSX_DEPLOYMENT_TARGET in .github/workflows/wheels.yml.
    static_dir="$workdir/static-fortran-runtime"
    mkdir -p "$static_dir"
    for lib in libgfortran.a libquadmath.a libgcc.a libgcc_eh.a libemutls_w.a; do
      archive="$(gfortran -print-file-name="$lib")"
      if [ -f "$archive" ]; then
        cp "$archive" "$static_dir/"
      fi
    done
    export LDFLAGS="-L$static_dir ${LDFLAGS:-}"
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    ;;

  MINGW*|MSYS*)
    # MSYS2 UCRT64 shell on a Windows runner. gcc/gfortran, reference
    # LAPACK/BLAS and pkgconf come from pacman (installed by the workflow's
    # setup-msys2 step). Use a native (C:/...) prefix: setup.py later runs
    # pkg-config from a plain Windows process, where MSYS2's /c/... path
    # translation is not available, so the paths recorded in the .pc files
    # must be native.
    PREFIX="C:/ipopt"
    LAPACK_LFLAGS="-L$(cygpath -m "$MSYSTEM_PREFIX/lib") -llapack -lblas"
    # No PKG_CONFIG_PATH here: path(-list) conversion between the MSYS2
    # shell and the native pkgconf is unreliable ("C:" contains ':', the
    # unix separator), which made Ipopt's configure silently skip MUMPS.
    # MUMPS is passed to configure explicitly below instead; setup.py
    # (running outside MSYS2) gets its own PKG_CONFIG_PATH via
    # pyproject.toml [tool.cibuildwheel.windows].
    ;;

  *)
    echo "Unsupported platform: $(uname)" >&2
    exit 1
    ;;
esac

# MUMPS, via coin-or's autotools wrapper:
curl -fsSL "https://github.com/coin-or-tools/ThirdParty-Mumps/archive/refs/tags/releases/${COIN_MUMPS_VERSION}.tar.gz" | tar xz
cd "ThirdParty-Mumps-releases-${COIN_MUMPS_VERSION}"
./get.Mumps
./configure --prefix="$PREFIX" --with-lapack="$LAPACK_LFLAGS"
make -j"$NPROC"
make install
cd ..

# Ipopt itself (no Fortran needed here). --disable-java: the runners have a
# JDK which configure would otherwise pick up. MUMPS (installed above) is
# passed explicitly instead of via pkg-config: a failing pkg-config lookup
# (e.g. inside MSYS2) makes configure silently skip MUMPS, yielding an Ipopt
# without any linear solver whose solve() aborts before the first iteration.
curl -fsSL "https://github.com/coin-or/Ipopt/archive/refs/tags/releases/${IPOPT_VERSION}.tar.gz" | tar xz
cd "Ipopt-releases-${IPOPT_VERSION}"
./configure --prefix="$PREFIX" \
  --with-lapack="$LAPACK_LFLAGS" \
  --with-mumps-cflags="-I$PREFIX/include/coin-or/mumps" \
  --with-mumps-lflags="-L$PREFIX/lib -lcoinmumps" \
  --disable-java
# Guard against any remaining silent degradation:
grep -qs "#define IPOPT_HAS_MUMPS 1" src/Common/config.h config.h || {
  echo "ERROR: Ipopt configured itself without MUMPS (no linear solver)." >&2
  echo "Check config.log for the failed MUMPS compile/link test." >&2
  exit 1
}
make -j"$NPROC"
make install
