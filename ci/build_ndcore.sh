#!/bin/bash
#  - Blue Brain Project -
# This script builds the mod extensions to neurodamus. The folder gets named _lib
build_common()
{
    cp -fL $CORE_DIR/mod/*.mod $MOD_DIR
    cp -f $MECHS_DIR/common/*.mod $MOD_DIR
    cd $BUILD_DIR
    nrnivmodl -coreneuron -incflags "-DDISABLE_REPORTINGLIB" $MOD_DIR   
}

build_common_ngv()
{
    cp -fL $CORE_DIR/mod/*.mod $MOD_DIR
    cp -f $MECHS_DIR/ngv/*.mod $MOD_DIR
    cd $BUILD_DIR
    nrnivmodl -incflags "-DDISABLE_REPORTINGLIB" $MOD_DIR
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

CORE_DIR="$1"
MECHS_DIR="$2"
BUILD_DIR="$3"

if [ $NGV_BUILD = true ]; then 
    LIBRARY_DIR=$BUILD_DIR/lib-ngv
    MOD_DIR=$BUILD_DIR/mods-ngv.tmp
elif [ $ALLEN_V1_BUILD = true ]; then
    LIBRARY_DIR=$BUILD_DIR/lib-allen-v1
    MOD_DIR=$BUILD_DIR/mods-allen-v1.tmp
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

#Build libs from mod files
mkdir -p $MOD_DIR
mkdir -p $LIBRARY_DIR
if [ $NGV_BUILD = true ]; then
    build_common_ngv
else
    build_common
fi

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
