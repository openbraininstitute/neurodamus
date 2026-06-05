#!/usr/bin/env bash
# Install system-level APT packages required for building (compilers, flex, bison, etc.).
# Note; this doesn't include MPI - some environments supply their own MPI
#
# Environment variables:
#   SUDO Set to "sudo" if elevated privileges are needed (e.g. SUDO=sudo)

install-apt-dependencies() {
    $SUDO apt-get install --yes \
        --no-install-suggests \
        --no-install-recommends \
        g++ \
        gcc \
        make \
        git \
        wget \
        flex \
        libfl-dev \
        bison \
        ninja-build \
        libreadline-dev \
        curl \
        ca-certificates \
        pkg-config \
        jq
}
