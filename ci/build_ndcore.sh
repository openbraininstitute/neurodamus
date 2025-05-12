#!/bin/bash
#  - Blue Brain Project -
# This script builds the mod extensions to neurodamus. The folder gets named _lib
build_common()
{
    cp -f $CORE_DIR/mod/*.mod $MOD_DIR
    cp -f $NEURODAMUS_MODELS_DIR/common/mod/*.mod $MOD_DIR
    cd $BUILD_DIR
    nrnivmodl -coreneuron -incflags "-DDISABLE_REPORTINGLIB" $MOD_DIR   
}

build_common_ngv()
{
    cp -f $CORE_DIR/mod/*.mod $MOD_DIR
    cp -f $NEURODAMUS_MODELS_DIR/common/mod/*.mod $MOD_DIR
    cp -f $NEURODAMUS_MODELS_DIR/common/mod/ngv/*.mod $MOD_DIR
    cd $BUILD_DIR
    nrnivmodl -incflags "-DDISABLE_REPORTINGLIB" $MOD_DIR
}

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

# Check if library already exists and export ENV variables in that case
if [ $(uname) = "Darwin" ]; then EXT="dylib"; else EXT="so"; fi
if [ -f "$LIBRARY_DIR/libnrnmech.$EXT" ]; then
    echo "export HOC_LIBRARY_PATH=$LIBRARY_DIR" > $BUILD_DIR/.envfile
    echo "export NRNMECH_LIB_PATH=$LIBRARY_DIR/libnrnmech.$EXT" >> $BUILD_DIR/.envfile
    if [ -f "$LIBRARY_DIR/libcorenrnmech.$EXT" ]; then
        echo "export CORENEURONLIB=$LIBRARY_DIR/libcorenrnmech.$EXT" >> $BUILD_DIR/.envfile
    fi
    exit 0
fi


# Get the common synapses and mechanisms 
NEURODAMUS_MODELS_DIR=$BUILD_DIR/neurodamus-models
if [ -d "$NEURODAMUS_MODELS_DIR" ]; then
    ( cd "$NEURODAMUS_MODELS_DIR" && git pull --quiet )
else
    git clone https://github.com/openbraininstitute/neurodamus-models.git $NEURODAMUS_MODELS_DIR --depth=1
fi

#Build libs from mod files
mkdir -p $MOD_DIR
mkdir -p $LIBRARY_DIR
if [ $NGV_BUILD = true ]; then build_common_ngv; else build_common; fi
ARCH=$(uname -m)
if [ ! -f $ARCH/special ]; then
    echo "Error running nrnivmodl"
    exit 1
fi

#Copy and export libraries
cp -f $ARCH/libnrnmech.$EXT $LIBRARY_DIR
echo "export HOC_LIBRARY_PATH=$LIBRARY_DIR" > $BUILD_DIR/.envfile
echo "export NRNMECH_LIB_PATH=$LIBRARY_DIR/libnrnmech.$EXT" >> $BUILD_DIR/.envfile
if [ -f "$ARCH/libcorenrnmech.$EXT" ]; then
    cp -f $ARCH/libcorenrnmech.$EXT $LIBRARY_DIR
    echo "export CORENEURONLIB=$LIBRARY_DIR/libcorenrnmech.$EXT" >> $BUILD_DIR/.envfile
fi
cp -f $CORE_DIR/hoc/*.hoc $LIBRARY_DIR
cp -f $NEURODAMUS_MODELS_DIR/common/hoc/*.hoc $LIBRARY_DIR
