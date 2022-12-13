FROM quay.io/pypa/manylinux2014_x86_64

ARG LAPACK_VERSION=3.10.1
ARG IPOPT_VERSION=3.12.12

# Install lapack (reference), including blas:
RUN curl -L https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v${LAPACK_VERSION}.tar.gz | tar --no-same-owner -xzf - \
  && cmake -S lapack-${LAPACK_VERSION} -B build -DCBLAS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_LIBDIR=/usr/local/lib \
  && cmake --build build --target install \
  && rm -rf build lapack-${LAPACK_VERSION}

# Install mumps:
RUN yum install -y MUMPS MUMPS-devel
# Install ipopt:
RUN curl "https://www.coin-or.org/download/source/Ipopt/Ipopt-${IPOPT_VERSION}.tgz" | tar -xzf - \
  && mv "Ipopt-${IPOPT_VERSION}" ipopt \
  && mkdir -p ipopt_build \
  && cd ipopt_build \
  && ADD_CFLAGS=-lrt ../ipopt/configure --enable-static --prefix=/usr/local --with-mumps-incdir="/usr/include/MUMPS" --with-mumps-lib="-lmumps_common -ldmumps -lzmumps -lsmumps -lcmumps" \
  && make \
  && make install \
  && cd - \
  && rm -rf ipopt ipopt_build

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
