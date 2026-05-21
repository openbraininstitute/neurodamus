#!/bin/bash
# Install Python build dependencies (cython, numpy, cmake, mpi4py, etc.).
#
# Environment variables:
#   PRE              called on function entry
#   PIP pip command to use (e.g. "uv pip")

install-python-dependencies() {
    PRE || true

    $PIP install --upgrade pip setuptools
    $PIP install cython numpy wheel pkgconfig jinja2 pyyaml cmake
    $PIP install mpi4py openmpi
}
