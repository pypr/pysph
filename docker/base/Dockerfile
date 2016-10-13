FROM ubuntu:16.04
MAINTAINER Prabhu Ramachandran <prabhu@aero.iitb.ac.in>

# Install the necessary packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cython \
    cython3 \
    g++ \
    git \
    ipython \
    ipython3 \
    libgomp1 \
    libopenmpi-dev \
    libtrilinos-zoltan-dev \
    mayavi2 \
    python \
    python-dev \
    python-execnet \
    python-h5py \
    python-mako \
    python-matplotlib \
    python-mock \
    python-mpi4py \
    python-nose \
    python-numpy \
    python-pip \
    python-psutil \
    python-qt4 \
    python-setuptools \
    python-unittest2 \
    python3 \
    python3-h5py \
    python3-mako \
    python3-matplotlib \
    python3-mpi4py \
    python3-nose \
    python3-numpy \
    python3-pip \
    python3-psutil \
    sudo \
    tox \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Sudo and the new user are needed as one should not run mpiexec as root.
RUN adduser --disabled-password --gecos '' pysph && \
    adduser pysph sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ENV HOME=/home/pysph \
    ZOLTAN_INCLUDE=/usr/include/trilinos \
    ZOLTAN_LIBRARY=/usr/lib/x86_64-linux-gnu \
    USE_TRILINOS=1

USER pysph
VOLUME /home/pysph
WORKDIR /home/pysph
