FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ARG LIBSONATA_TAG=master
ARG LIBSONATAREPORT_TAG=master
ARG NEURON_TAG=master
ARG NEURON_COMMIT_ID

RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && apt-get --yes \
        -qq \
        --no-install-suggests \
        --no-install-recommends \
        install \
        curl \
        ca-certificates \
        jq \
        make \
        g++ \
        gcc \
        python3 \
        python3-dev \
        git \
        cmake \
        wget \
        vim \
        mpich \
        libmpich-dev \
        libhdf5-mpich-dev \
        hdf5-tools \
        flex \
        libfl-dev \
        bison \
        ninja-build \
        libreadline-dev

COPY build-dependencies.sh /opt/

RUN source /opt/build-dependencies.sh \
 && install-uv

RUN source /opt/build-dependencies.sh \
 && make-venv

RUN source /opt/build-dependencies.sh \
 && source $VENV_PATH/bin/activate \
 && install-python-dependencies

RUN source /opt/build-dependencies.sh \
 && source $VENV_PATH/bin/activate \
 && build-h5py

RUN source /opt/build-dependencies.sh \
 && source $VENV_PATH/bin/activate \
 && LIBSONATA_LATEST=`github-tag openbraininstitute/libsonata` \
 && build-libsonata $LIBSONATA_LATEST

RUN source /opt/build-dependencies.sh \
 && source $VENV_PATH/bin/activate \
 && LIBSONATA_REPORT_LATEST=`github-tag openbraininstitute/libsonatareport` \
 && build-libsonatareport $LIBSONATA_REPORT_LATEST

RUN source /opt/build-dependencies.sh \
 && source $VENV_PATH/bin/activate \
 && NEURON_BRANCH=master \
 && NEURON_COMMIT_ID=0d990513b \
 && build-neuron $NEURON_BRANCH $NEURON_COMMIT_ID

RUN source /opt/build-dependencies.sh \
 && source $VENV_PATH/bin/activate \
 && install-neurodamus

RUN source /opt/build-dependencies.sh \
 && source $VENV_PATH/bin/activate \
 && NEURODAMUS_MODELS_LATEST=`github-tag openbraininstitute/neurodamus-models` \
 && build-neocortex-models $NEURODAMUS_MODELS_LATEST
