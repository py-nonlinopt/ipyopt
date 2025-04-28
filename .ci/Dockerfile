FROM quay.io/pypa/manylinux2014_x86_64

ARG LAPACK_VERSION=3.10.1
ARG IPOPT_VERSION=3.12.12
ARG COIN_OR_MUMPS_VERSION=3.0.9

# Install lapack (reference), including blas:
RUN curl -L https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v${LAPACK_VERSION}.tar.gz | tar --no-same-owner -xzf - \
  && cmake -S lapack-${LAPACK_VERSION} -B build -DCBLAS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_LIBDIR=/usr/local/lib \
  && cmake --build build --target install \
  && rm -rf build lapack-${LAPACK_VERSION}

# Install mumps:
RUN yum install -y gcc-gfortran \
  && curl -L https://github.com/coin-or-tools/ThirdParty-Mumps/archive/refs/tags/releases/${COIN_OR_MUMPS_VERSION}.tar.gz | tar --no-same-owner -xzf - \
  && cd ThirdParty-Mumps-releases* \
  && ./get.Mumps \
  && ./configure \
  && make \
  && make install

# Install ipopt:
RUN curl "https://www.coin-or.org/download/source/Ipopt/Ipopt-${IPOPT_VERSION}.tgz" | tar -xzf - \
  && mv "Ipopt-${IPOPT_VERSION}" ipopt \
  && sed -i -e 's/#include "mpi.h"/#include "mumps_mpi.h"/' ipopt/Ipopt/src/Algorithm/LinearSolvers/IpMumpsSolverInterface.cpp \
  && mkdir -p ipopt_build \
  && cd ipopt_build \
  && ADD_CFLAGS=-lrt ../ipopt/configure --enable-static --prefix=/usr/local \
  && make \
  && make install \
  && cd - \
  && rm -rf ipopt ipopt_build

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
