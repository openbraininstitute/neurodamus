#!/usr/bin/env bash
# Generate an environment activation script (env-neocortex.sh) for running neurodamus.
#
# Environment variables:
#   INSTALL_DIR   - Prefix where everything is installed (e.g. /opt/obi)
#   VIRTUAL_ENV   - Path to the active Python virtual environment

make-env() {
    : "${INSTALL_DIR:?INSTALL_DIR is not set}"

    local ARCH=$(uname -m)
    local LIB_EXT=$(if [[ $(uname) == Darwin ]]; then echo dylib; else echo so; fi)
    local NEURODAMUS_PYTHON=$(python3 -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")
    local NEURODAMUS_NEOCORTEX_ROOT=$INSTALL_DIR/neurodamus-models

    cat > $INSTALL_DIR/env-neocortex.sh << _EOF
    export NEURODAMUS_NEOCORTEX_ROOT=$NEURODAMUS_NEOCORTEX_ROOT
    source $VIRTUAL_ENV/bin/activate
    export NEURODAMUS_PYTHON=$NEURODAMUS_PYTHON
    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/hoc
    export CORENEURONLIB=$INSTALL_DIR/$ARCH/libcorenrnmech.$LIB_EXT
    export NRNMECH_LIB_PATH=$INSTALL_DIR/$ARCH/libnrnmech.$LIB_EXT
    export PATH=$INSTALL_DIR/$ARCH:$PATH
_EOF
}
