#!/bin/bash

# export CMAKE_BUILD_TYPE=RelWithDebugInfo
# export BUILD_DIR=/tmp
# export INSTALL_DIR=/opt/obi
# PIP='uv pip' build-libsonata v0.1.35

build-libsonata() {
    pre

    : "${PIP:?PIP is not set}"

    local COMMIT=${1:-HEAD}

    local LIBSONATA=$BUILD_DIR/libsonata

    git clone --filter=blob:none \
        https://github.com/openbraininstitute/libsonata \
        $LIBSONATA

    (cd $LIBSONATA && \
        git fetch --depth 1 origin $COMMIT &&
        git checkout FETCH_HEAD &&
        git submodule update --init --depth=1 \
        extlib/HighFive \
        extlib/fmt \
        python/pybind11
    )

    if [[ -n $SCCACHE_DIR ]]; then
        echo "Using sccache"
        export CC="sccache mpicc"
        export CXX="sccache mpic++"
    else
        export CC="mpicc"
        export CXX="mpic++"
    fi

    CMAKE_PREFIX_PATH=$INSTALL_DIR \
    CMAKE_GENERATOR=Ninja \
    SONATA_BUILD_TYPE=$CMAKE_BUILD_TYPE \
    $PIP -v install $LIBSONATA
}
