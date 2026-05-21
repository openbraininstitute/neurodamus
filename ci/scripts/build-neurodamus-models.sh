#!/bin/bash
# Build neocortex neurodamus-models.
#
# Environment variables:
#   PRE         called on function entry
#   BUILD_DIR   Directory for cloning/building source (e.g. /tmp)
#   INSTALL_DIR Prefix where dependencies are installed and models will be installed

build-neocortex-models() {
    PRE || true

    local COMMIT=${1:-HEAD}

    local NEOCORTEX_MOD=$BUILD_DIR/neurodamus-models/
    local NEOCORTEX_MOD_BUILD=$NEOCORTEX_MOD/build
    local NEOCORTEX_MOD_INSTALL=$INSTALL_DIR/neurodamus-models

    export PATH=$INSTALL_DIR/bin:$PATH

    if [[ ! -e $NEOCORTEX_MOD ]]; then
        git clone --filter=blob:none \
	    https://github.com/openbraininstitute/neurodamus-models.git \
	    $NEOCORTEX_MOD
    fi

    ( cd $NEOCORTEX_MOD && \
        git fetch --depth 1 origin $COMMIT &&
        git checkout FETCH_HEAD
    )

    DATADIR=$(python3 -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

    cmake -B $NEOCORTEX_MOD_BUILD -S $NEOCORTEX_MOD  \
      -GNinja \
      -DCMAKE_INSTALL_PREFIX=$NEOCORTEX_MOD_INSTALL \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_PREFIX_PATH=$INSTALL_DIR \
      -DNEURODAMUS_CORE_DIR=${DATADIR} \
      -DNEURODAMUS_MECHANISMS=neocortex \
      -DNEURODAMUS_NCX_V5=ON

    cmake --build $NEOCORTEX_MOD_BUILD
    cmake --install $NEOCORTEX_MOD_BUILD
}
