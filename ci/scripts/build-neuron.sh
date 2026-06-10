#!/usr/bin/env bash
# Build and install NEURON simulator from source with CoreNEURON and MPI support.
# Assumes libsonatareport has already been installed in $INSTALL_DIR.
#
# Environment variables:
#   PRE              called on function entry
#   BUILD_DIR        Directory for cloning/building source (e.g. /tmp)
#   INSTALL_DIR      Prefix where dependencies are installed (e.g. /opt/obi)
#   CMAKE_BUILD_TYPE CMake build type (e.g. RelWithDebugInfo)
#   PIP              pip command to use (e.g. "uv pip")
#   SCCACHE_DIR      If set, enables sccache for compilation

build-neuron() {
    PRE || true

    : "${PIP:?PIP is not set}"

    local COMMIT=${1:-HEAD}

    local NRN=$BUILD_DIR/nrn
    local NRN_BUILD=$NRN/build

    if [[ ! -e $NRN ]]; then
        git clone --filter=blob:none \
            https://github.com/neuronsimulator/nrn.git $NRN
    fi

    ( cd $NRN && \
        git fetch --depth 1 origin $COMMIT &&
        git checkout FETCH_HEAD &&
        git submodule update --init --depth=1 --recursive \
            external/spdlog \
            external/Random123 \
            external/fmt \
            external/nanobind \
    )

    $PIP install tomli tomli_w
    grep -q neuron-nightly $NRN/pyproject.toml && python3 $NRN/packaging/python/change_name.py $NRN/pyproject.toml neuron

    local CMAKE_ARGS=(
      -DNRN_ENABLE_MPI_DYNAMIC=OFF
      -DNRN_ENABLE_MPI=ON
      -DNRN_ENABLE_RX3D=OFF
      -DNRN_ENABLE_INTERVIEWS=OFF
      -DNRN_ENABLE_CORENEURON=ON
      -DNMODL_ENABLE_PYTHON_BINDINGS=OFF
      -DCORENRN_ENABLE_REPORTING=ON
      -DCMAKE_PREFIX_PATH=$INSTALL_DIR
      -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
    )

    if [[ $(uname) == Darwin ]]; then
        CMAKE_ARGS+=(
            -DCMAKE_C_COMPILER=gcc
            -DCMAKE_CXX_COMPILER=g++
        )

        # NRN: workaround for fmt 11.1 (see https://github.com/gabime/spdlog/pull/3312)
        brew unlink fmt
        export PATH="$(brew --prefix)/opt/flex/bin:$(brew --prefix)/opt/bison/bin":$PATH
    fi

    if [[ -n $SCCACHE_DIR ]]; then
        CMAKE_ARGS+=(
            -DCMAKE_C_COMPILER_LAUNCHER=sccache
            -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
        )
    fi

    CMAKE_ARGS="${CMAKE_ARGS[@]}" \
      $PIP install -v $NRN

    if [[ -n $SCCACHE_DIR ]]; then
        sccache --show-stats
    fi
}
