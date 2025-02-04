#!/bin/bash
#  - Blue Brain Project -
# This script builds the mod extensions to neurodamus. The folder gets named _lib

set -euxo pipefail

CORE_DIR="$1"
BUILD_DIR="$2"

LIBRARY_DIR=$BUILD_DIR/lib
if [ $(uname) = "Darwin" ]; then EXT="dylib"; else EXT="so"; fi
if [ -f "$LIBRARY_DIR/libnrnmech.$EXT" ]; then
    echo "export HOC_LIBRARY_PATH=$LIBRARY_DIR" > $BUILD_DIR/.envfile
    echo "export NRNMECH_LIB_PATH=$LIBRARY_DIR/libnrnmech.$EXT" >> $BUILD_DIR/.envfile
    exit 0
fi


# Get the common synapses
NEURODAMUS_MODELS_DIR=$BUILD_DIR/neurodamus-models
if [ -d "$NEURODAMUS_MODELS_DIR" ]; then
    ( cd "$NEURODAMUS_MODELS_DIR" && git pull --quiet )
else
    git clone -b katta/conn_conf_remove https://github.com/openbraininstitute/neurodamus-models.git $NEURODAMUS_MODELS_DIR --depth=1
    # git clone https://github.com/openbraininstitute/neurodamus-models.git $NEURODAMUS_MODELS_DIR --depth=1
fi

MOD_DIR=$BUILD_DIR/mods.tmp
mkdir -p $MOD_DIR
cp -f $CORE_DIR/mod/*.mod $MOD_DIR
cp -f $NEURODAMUS_MODELS_DIR/common/mod/*.mod $MOD_DIR
cd $BUILD_DIR
nrnivmodl -incflags "-DDISABLE_REPORTINGLIB -DDISABLE_HDF5 -DDISABLE_MPI" $MOD_DIR
ARCH=$(uname -m)
if [ ! -f $ARCH/special ]; then
    echo "Error running nrnivmodl"
    exit 1
fi

mkdir -p $LIBRARY_DIR
cp -f $ARCH/libnrnmech.$EXT $LIBRARY_DIR
echo "export HOC_LIBRARY_PATH=$LIBRARY_DIR" > $BUILD_DIR/.envfile
echo "export NRNMECH_LIB_PATH=$LIBRARY_DIR/libnrnmech.$EXT" >> $BUILD_DIR/.envfile

cp -f $CORE_DIR/hoc/*.hoc $LIBRARY_DIR
cp -f $NEURODAMUS_MODELS_DIR/common/hoc/*.hoc $LIBRARY_DIR
echo $1
ls -l $LIBRARY_DIR
