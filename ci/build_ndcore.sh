#!/bin/bash
#  - Blue Brain Project -
# This script builds the mod extensions to neurodamus. The folder gets named _lib

set -euxo pipefail

CORE_DIR="$1"

if [ -d _lib ]; then
    exit 0
fi

# Get the common synapses
COMMON_DIR=neurodamus-models/common
if [ -d "$COMMON_DIR" ]; then
    ( cd "$COMMON_DIR" && git pull --quiet )
else
    git clone https://github.com/openbraininstitute/neurodamus-models.git neurodamus-models  --depth=1
fi

MOD_BUILD_DIR="mods.tmp"
mkdir -p $MOD_BUILD_DIR
cp -f $CORE_DIR/mod/*.mod $MOD_BUILD_DIR
cp -f $COMMON_DIR/mod/*.mod $MOD_BUILD_DIR
nrnivmodl -incflags "-DDISABLE_REPORTINGLIB -DDISABLE_HDF5 -DDISABLE_MPI" $MOD_BUILD_DIR
ARCH=$(uname -m)
if [ ! -f $ARCH/special ]; then
    echo "Error running nrnivmodl"
    exit 1
fi
mkdir -p _lib
cp -f $ARCH/libnrnmech* _lib/
cp -f $CORE_DIR/hoc/*.hoc _lib/
cp -f $COMMON_DIR/hoc/*.hoc _lib/

echo $1
ls -l
ls -l _lib
