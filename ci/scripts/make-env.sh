#!/bin/bash
# export INSTALL_DIR=/opt/obi

make-env() {
    : "${INSTALL_DIR:?INSTALL_DIR is not set}"

    local NEURODAMUS_PYTHON=$(python3 -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")
    local NEURODAMUS_NEOCORTEX_ROOT=$INSTALL_DIR/neurodamus-models

    cat > $INSTALL_DIR/env-neocortex.sh << _EOF
    source $VIRTUAL_ENV/bin/activate
    export NEURODAMUS_PYTHON=$NEURODAMUS_PYTHON
    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc
    export CORENEURONLIB=$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so
    export NRNMECH_LIB_PATH=$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so
    export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH
_EOF
ls -al $INSTALL_DIR
}
