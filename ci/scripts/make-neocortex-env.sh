#!/usr/bin/env bash
# Generate an environment activation script (env-neocortex.sh) for running neurodamus.
#
# Environment variables:
#   INSTALL_DIR   - Prefix where everything is installed (e.g. /opt/obi)
#   VIRTUAL_ENV   - Path to the active Python virtual environment
#   BASE_DIR      - Neurodamus source base

make-neocortex-env() {
    : "${INSTALL_DIR:?INSTALL_DIR is not set}"

    local NEURODAMUS_PYTHON=$(python3 -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

    local CMD=(
        neurodamus-compile-mods
        --input-dir $BASE_DIR/tests/mechanisms/neocortex
        --output-dir $INSTALL_DIR/neocortex
        --with-internal-mods
        --output-type shell
        --simulator=coreneuron
    )
    exports=$(${CMD[@]})

    while IFS= read -r line; do
        export "$line"
    done <<< "$exports"

    cat > $INSTALL_DIR/env-neocortex.sh << _EOF
source $VIRTUAL_ENV/bin/activate

export NEURODAMUS_PYTHON=$NEURODAMUS_PYTHON
export HOC_LIBRARY_PATH=$NEURODAMUS_PYTHON/hoc
export NRNMECH_LIB_PATH=$NRNMECH_LIB_PATH
export CORENEURONLIB=$CORENEURONLIB
export PATH=$SPECIALS_PATH:\$PATH
_EOF
}
