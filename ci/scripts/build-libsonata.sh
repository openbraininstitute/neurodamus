#!/bin/bash
# Build and install libsonata
#
# Environment variables:
#   PRE              called on function entry
#   BUILD_DIR        Directory for cloning/building source (e.g. /tmp)
#   INSTALL_DIR      Prefix where dependencies are installed (e.g. /opt/obi)
#   CMAKE_BUILD_TYPE CMake build type (e.g. RelWithDebugInfo)
#   PIP              pip command to use (e.g. "uv pip")

build-libsonata() {
    PRE || true

    : "${PIP:?PIP is not set}"

    local COMMIT=${1:-HEAD}

    local LIBSONATA=$BUILD_DIR/libsonata

    [[ -e $LIBSONATA ]] || git clone --filter=blob:none \
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

    CC="mpicc" \
    CXX="mpic++" \
    CMAKE_PREFIX_PATH=$INSTALL_DIR \
    CMAKE_GENERATOR=Ninja \
    SONATA_BUILD_TYPE=$CMAKE_BUILD_TYPE \
    $PIP -v install $LIBSONATA
}
