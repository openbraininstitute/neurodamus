#!/usr/bin/env bash
# 
# Environment variables:
#   INSTALL_DIR   - Prefix where everything is installed (e.g. /opt/obi)
#   VIRTUAL_ENV   - Path to the active Python virtual environment

make-build-neurodamus-models() {
    set -x
    local LIB_EXT=$(if [[ $(uname) == Darwin ]]; then echo dylib; else echo so; fi)
    INCLUDE_FLAGS="'-isystem $INSTALL_DIR/include'"
    LOAD_FLAGS="'-Wl,-rpath,$INSTALL_DIR/lib $INSTALL_DIR/lib/libsonatareport.$LIB_EXT'"

    cat > $INSTALL_DIR/build-neurodamus-models.sh << _EOF
#!/bin/bash
echo "Build models from folder " \$1

source $VIRTUAL_ENV/bin/activate

if [[ "\$2" == "--only-neuron" ]]; then
    nrnivmodl -incflags $INCLUDE_FLAGS -loadflags $LOAD_FLAGS "\$1"
else
    nrnivmodl -coreneuron -incflags $INCLUDE_FLAGS -loadflags $LOAD_FLAGS "\$1"
fi
_EOF
    chmod +x $INSTALL_DIR/build-neurodamus-models.sh
}
