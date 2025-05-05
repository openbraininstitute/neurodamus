#!/bin/bash
#  - Blue Brain Project -
# This script builds the mod extensions to neurodamus. The folder gets named _lib

set -euxo pipefail

NGV_BUILD=false
remaining_args=""
for arg in "$@"; do
    if [[ "$arg" == -ngv ]]; then
        NGV_BUILD=true
    else
        remaining_args="$remaining_args $arg" 
    fi
done
set -- $remaining_args

CORE_DIR="$1"
BUILD_DIR="$2"

if [ $NGV_BUILD = true ]; then 
    LIBRARY_DIR=$BUILD_DIR/lib-ngv
    MOD_DIR=$BUILD_DIR/mods-ngv.tmp
else 
    LIBRARY_DIR=$BUILD_DIR/lib
    MOD_DIR=$BUILD_DIR/mods.tmp
fi

if [ $(uname) = "Darwin" ]; then EXT="dylib"; else EXT="so"; fi
if [ -f "$LIBRARY_DIR/libnrnmech.$EXT" ]; then
    echo "export HOC_LIBRARY_PATH=$LIBRARY_DIR" > $BUILD_DIR/.envfile
    echo "export NRNMECH_LIB_PATH=$LIBRARY_DIR/libnrnmech.$EXT" >> $BUILD_DIR/.envfile
    if [ -f "$LIBRARY_DIR/libcorenrnmech.$EXT" ]; then
        echo "export CORENEURONLIB=$LIBRARY_DIR/libcorenrnmech.$EXT" >> $BUILD_DIR/.envfile
    fi
    exit 0
fi

mkdir -p $MOD_DIR
mkdir -p $LIBRARY_DIR

# Get the common synapses
NEURODAMUS_MODELS_DIR=$BUILD_DIR/neurodamus-models
if [ -d "$NEURODAMUS_MODELS_DIR" ]; then
    ( cd "$NEURODAMUS_MODELS_DIR" && git pull --quiet )
else
    git clone https://github.com/openbraininstitute/neurodamus-models.git $NEURODAMUS_MODELS_DIR --depth=1
fi


build_common()
{
    cp -f $CORE_DIR/mod/*.mod $MOD_DIR
    cp -f $NEURODAMUS_MODELS_DIR/common/mod/*.mod $MOD_DIR
    cd $BUILD_DIR
    nrnivmodl -coreneuron -incflags "-DDISABLE_REPORTINGLIB" $MOD_DIR
    ARCH=$(uname -m)
    if [ ! -f $ARCH/special ]; then
        echo "Error running nrnivmodl"
        exit 1
    fi

    cp -f $ARCH/libnrnmech.$EXT $LIBRARY_DIR
    cp -f $ARCH/libcorenrnmech.$EXT $LIBRARY_DIR
    echo "export HOC_LIBRARY_PATH=$LIBRARY_DIR" > $BUILD_DIR/.envfile
    echo "export NRNMECH_LIB_PATH=$LIBRARY_DIR/libnrnmech.$EXT" >> $BUILD_DIR/.envfile
    echo "export CORENEURONLIB=$LIBRARY_DIR/libcorenrnmech.$EXT" >> $BUILD_DIR/.envfile

    cp -f $CORE_DIR/hoc/*.hoc $LIBRARY_DIR
    cp -f $NEURODAMUS_MODELS_DIR/common/hoc/*.hoc $LIBRARY_DIR
}


build_common_ngv()
{
    cp -f $CORE_DIR/mod/*.mod $MOD_DIR
    cp -f $NEURODAMUS_MODELS_DIR/common/mod/*.mod $MOD_DIR
    cp -f $NEURODAMUS_MODELS_DIR/common/mod/ngv/*.mod $MOD_DIR
    cd $BUILD_DIR
    nrnivmodl -incflags "-DDISABLE_REPORTINGLIB" $MOD_DIR
    ARCH=$(uname -m)
    if [ ! -f $ARCH/special ]; then
        echo "Error running nrnivmodl"
        exit 1
    fi

    cp -f $ARCH/libnrnmech.$EXT $LIBRARY_DIR
    echo "export HOC_LIBRARY_PATH=$LIBRARY_DIR" > $BUILD_DIR/.envfile
    echo "export NRNMECH_LIB_PATH=$LIBRARY_DIR/libnrnmech.$EXT" >> $BUILD_DIR/.envfile

    cp -f $CORE_DIR/hoc/*.hoc $LIBRARY_DIR
    cp -f $NEURODAMUS_MODELS_DIR/common/hoc/*.hoc $LIBRARY_DIR    
}

if [ $NGV_BUILD = true ]; then build_common_ngv; else build_common; fi
