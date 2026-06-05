#!/usr/bin/env bash
# Build neocortex neurodamus-models.
#
# Args:
#   NEOCORTEX_MOD: path to the folder which contains the model mod files
#
# Environment variables:
#   PRE         called on function entry
#   BUILD_DIR   Directory for cloning/building source (e.g. /tmp)
#   INSTALL_DIR Prefix where dependencies are installed and models will be installed
#   BASE_DIR    Base directory of this repository

build-neocortex-models() {
    PRE || true

    local NEOCORTEX_MOD=$1

    export PATH=$INSTALL_DIR/bin:$PATH

    DATADIR=$(python3 -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

    pushd $INSTALL_DIR/
    mkdir -p neurodamus-models/mod
    mkdir -p neurodamus-models/hoc
    cp $DATADIR/mod/*.mod neurodamus-models/mod/
    cp $DATADIR/mod/*.hoc neurodamus-models/hoc/
    cp $NEOCORTEX_MOD/*.mod neurodamus-models/mod/
    source $BASE_DIR/ci/scripts/make-build-neurodamus-models.sh
    ./$INSTALL_DIR/build-neurodaus-models.sh neurodamus-models/mod
    popd

}
