#!/bin/bash
# export BUILD_DIR=/tmp
# export INSTALL_DIR=/opt/obi
# export CMAKE_BUILD_TYPE=RelWithDebugInfo
# PIP='uv pip'

# Note: this assumes that libsonatareports has been installed in $INSTALL_DIR

build-neuron() {
    pre || true

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
    python3 $NRN/packaging/python/change_name.py $NRN/pyproject.toml neuron

    #XXX
      #-DPYTHON_EXECUTABLE=`which python3`
    local CMAKE_ARGS=(
      -DNRN_ENABLE_MPI_DYNAMIC=OFF
      -DNRN_ENABLE_MPI=ON
      -DNRN_ENABLE_RX3D=OFF
      -DNRN_ENABLE_INTERVIEWS=OFF
      -DNRN_ENABLE_CORENEURON=ON
      -DNMODL_ENABLE_PYTHON_BINDINGS=OFF
      -DCORENRN_ENABLE_REPORTING=ON
      -DCMAKE_PREFIX_PATH=$INSTALL_DIR
    )

    if [[ -n $SCCACHE_DIR ]]; then
        CMAKE_ARGS+=(
            -DCMAKE_C_COMPILER_LAUNCHER=sccache
            -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
        )
    fi

    CMAKE_ARGS="${CMAKE_ARGS[@]}" $PIP install -v $NRN

    if [[ -n $SCCACHE_DIR ]]; then
        sccache --show-stats
    fi
}
