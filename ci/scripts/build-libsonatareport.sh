#!/usr/bin/env bash
# Build and install libsonatareport
#
# Environment variables:
#   PRE              called on function entry
#   BUILD_DIR        Directory for cloning/building source (e.g. /tmp)
#   INSTALL_DIR      Prefix where the library will be installed (e.g. /opt/obi)
#   CMAKE_BUILD_TYPE CMake build type (e.g. RelWithDebugInfo)
#   SCCACHE_DIR      If set, enables sccache for compilation

build-libsonatareport() {
    PRE || true

    local COMMIT=${1:-HEAD}

    local LIBSONATAREPORT=$BUILD_DIR/libsonatareport/
    local LIBSONATAREPORT_BUILD=$LIBSONATAREPORT/build

    [[ -e $LIBSONATAREPORT ]] || \
        git clone --filter=blob:none \
            https://github.com/openbraininstitute/libsonatareport.git \
            $LIBSONATAREPORT

    (cd $LIBSONATAREPORT && \
        git fetch --depth 1 origin $COMMIT &&
        git checkout FETCH_HEAD &&
        git submodule update --init --depth=1 \
        extlib/spdlog \
    )

    local CMAKE_ARGS=(
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
        -DSONATA_REPORT_ENABLE_SUBMODULES=ON
        -DSONATA_REPORT_ENABLE_MPI=ON
        -DSONATA_REPORT_ENABLE_TEST=OFF
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
        -DMPI_CXX_SKIP_MPICXX=ON
        -DCMAKE_CXX_FLAGS="-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX"
    )

    if [[ -n $SCCACHE_DIR ]]; then
        CMAKE_ARGS+=(
            -DCMAKE_C_COMPILER_LAUNCHER=sccache
            -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
        )
    fi

    cmake \
        -B $LIBSONATAREPORT_BUILD \
        -S $LIBSONATAREPORT \
        -G Ninja \
        "${CMAKE_ARGS[@]}"

    cmake --build $LIBSONATAREPORT_BUILD --parallel
    cmake --install $LIBSONATAREPORT_BUILD

    if [[ -n $SCCACHE_DIR ]]; then
        sccache --show-stats
    fi
}
