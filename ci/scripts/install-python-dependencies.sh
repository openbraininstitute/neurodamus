#!/usr/bin/env bash
# Install Python build dependencies
#
# Environment variables:
#   PRE              called on function entry
#   PIP pip command to use (e.g. "uv pip")

install-python-dependencies() {
    PRE || true

    $PIP install --upgrade pip setuptools
    $PIP install -v cython numpy wheel pkgconfig jinja2 pyyaml cmake ninja morphio
    $PIP install -v mpi4py
}
