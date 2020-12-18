FROM quay.io/pypa/manylinux1_x86_64

ARG BLAS_VERSION=3.8.0
ARG LAPACK_VERSION=3.4.2
ARG IPOPT_VERSION=3.12.11

# Notes:
# blas / lapack: need to pass -fPIC to enable dynamic linking

# Install blas (reference)
RUN curl http://www.netlib.org/blas/blas-${BLAS_VERSION}.tgz | tar -xzf - \
  && cd BLAS-${BLAS_VERSION} \
  && sed -i -e 's|^\(OPTS *=\).*$|\1 -O3 -fPIC|' -e 's|^\(NOOPT *=\).*$|\1 -O0 -fPIC|' -e 's|^\(BLASLIB *=\).*$|\1 libblas.a|' make.inc \
  && make \
  && gfortran -shared -o libblas.so *.o \
  && mv libblas.* /usr/local/lib/ \
  && cd - && rm -rf BLAS-${BLAS_VERSION}
# Install lapack (reference)
RUN curl http://www.netlib.org/lapack/lapack-${LAPACK_VERSION}.tgz | tar -xzf - \
  && cd lapack-${LAPACK_VERSION} \
  && sed -e 's|^\(OPTS *=\).*$|\1 -O2 -fPIC|' -e 's|^\(NOOPT *=\).*$|\1 -O0 -fPIC|' -e "s|^\(BLASLIB *= \).*$|\1/usr/local/lib/libblas.a|" make.inc.example > make.inc \
  && make lapacklib \
  && mv liblapack.a /usr/local/lib/ \
  && cd - && rm -rf lapack-${LAPACK_VERSION} \
  && mkdir /tmp/lapack && cd /tmp/lapack && ar -x /usr/local/lib/liblapack.a && gfortran -shared -o /usr/local/lib/liblapack.so /tmp/lapack/*.o && cd - && rm -rf /tmp/lapack
# Install IpOpt
# Notes: need ADD_CFLAGS=-lrt because of https://stackoverflow.com/questions/51209116/undefined-reference-to-clock-gettime-modify-makefile
RUN curl "https://www.coin-or.org/download/source/Ipopt/Ipopt-${IPOPT_VERSION}.tgz" | tar -xzf - \
  && mv "Ipopt-${IPOPT_VERSION}" ipopt \
  && echo -e '#!/bin/bash\n\
curl -L -O "$@"\n\
' > /usr/bin/wget \
  && chmod +x /usr/bin/wget \
  && for d in ASL Metis Mumps ; do cd "ipopt/ThirdParty/${d}" && ./get.${d} && cd - || exit 1 ; done \
  && mkdir -p ipopt_build \
  && cd ipopt_build \
  && ADD_CFLAGS=-lrt ../ipopt/configure --prefix=/usr/local \
  && make \
  && make install \
  && cd - \
  && rm -rf ipopt ipopt_build /usr/bin/wget
