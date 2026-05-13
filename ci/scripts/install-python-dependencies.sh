#!/bin/bash
#
# export PIP='uv pip'

install-python-dependencies() {
    pre || true

    $PIP install --upgrade pip setuptools
    $PIP install cython numpy wheel pkgconfig jinja2 pyyaml cmake
    $PIP install mpi4py openmpi
}
