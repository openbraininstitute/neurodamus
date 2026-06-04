#!/usr/bin/env bash
# Install system-level dnf packages required for building (compilers, MPI, flex, bison, etc.).
#
# Environment variables:
#   SUDO Set to "sudo" if elevated privileges are needed (e.g. SUDO=sudo)

install-dnf-dependencies() {
    $SUDO dnf -y $DNF_OPTIONS install \
        bison \
        gcc-c++ \
        flex \
        git \
        flex-devel \
        readline-devel
}
