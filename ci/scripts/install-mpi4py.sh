#!/usr/bin/env bash
# Build and install mpi4py
#
# Environment variables:
#   PRE              called on function entry
#   PIP              pip command to use (e.g. "uv pip")

install-mpi4py() {
    PRE || true

    $PIP install -v mpi4py
}
