#!/bin/bash
# export BUILD_DIR=/tmp
# export CMAKE_BUILD_TYPE=RelWithDebugInfo

build-libsonatareport() {
    pre || true

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
