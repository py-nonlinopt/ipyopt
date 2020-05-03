FROM debian:latest

ARG IPOPT_VERSION=3.12.11

RUN apt-get -qq update \
  && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    python3-wheel \
    twine \
    python3-dev \
    python3-numpy \
    python3-setuptools \
    build-essential \
    pkg-config \
    gfortran \
    liblapack-dev \
    libblas-dev \
  && rm -rf /var/lib/apt/lists/*

RUN wget -O - "https://www.coin-or.org/download/source/Ipopt/Ipopt-${IPOPT_VERSION}.tgz" | tar -xzf - \
  && mv "Ipopt-${IPOPT_VERSION}" ipopt \
  && for d in ASL Metis Mumps ; do cd "ipopt/ThirdParty/${d}" && ./get.${d} && cd - || exit 1 ; done \
  && mkdir -p "ipopt_build" \
  && cd "ipopt_build" \
  && ../ipopt/configure --prefix=/usr \
  && make \
  && make install \
  && cd - \
  && rm -rf ipopt ipopt_build
