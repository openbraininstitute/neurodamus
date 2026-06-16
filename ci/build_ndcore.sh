#!/bin/bash
#  - Open Brain Institute  -
# This script builds the mod extensions to neurodamus. The folder gets named _lib
build_mechanisms()
{
    cp -fL $CORE_DIR/mod/*.mod $MOD_DIR
    if [ -n "$MECHS_DIR" ]; then
        cp -f $MECHS_DIR/*.mod $MOD_DIR
    fi
    cd $BUILD_DIR
    nrnivmodl "${BUILD_OPT[@]}" $MOD_DIR
}

set -euxo pipefail

NGV_BUILD=false
ALLEN_V1_BUILD=false
remaining_args=""
for arg in "$@"; do
    if [[ "$arg" == -ngv ]]; then
        NGV_BUILD=true
    elif [[ "$arg" == -allen_v1 ]]; then
        ALLEN_V1_BUILD=true
    else
        remaining_args="$remaining_args $arg" 
    fi
done
set -- $remaining_args

BUILD_DIR="$1"
CORE_DIR="$2"
#Optional scientific mods
MECHS_DIR="${3:-}"
BUILD_OPT=("-incflags" "-DDISABLE_REPORTINGLIB")
if [ $NGV_BUILD = true ]; then 
    LIBRARY_DIR=$BUILD_DIR/lib-ngv
    MOD_DIR=$BUILD_DIR/mods-ngv.tmp
elif [ $ALLEN_V1_BUILD = true ]; then
    LIBRARY_DIR=$BUILD_DIR/lib-allen-v1
    MOD_DIR=$BUILD_DIR/mods-allen-v1.tmp
else
    LIBRARY_DIR=$BUILD_DIR/lib
    MOD_DIR=$BUILD_DIR/mods.tmp
    BUILD_OPT=("-coreneuron" "${BUILD_OPT[@]}")
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

#Build libs from mod files
mkdir -p $MOD_DIR
mkdir -p $LIBRARY_DIR
build_mechanisms

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
