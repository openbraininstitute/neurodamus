#!/bin/bash

install-h5py() {
    mkdir -p /tmp/h5py-stable
    export TMPDIR=/tmp/h5py-stable

    if [[ -e $INSTALL_DIR/include/H5Epublic.h ]]; then
        HDF5_INCLUDEDIR=$INSTALL_DIR/include/
        HDF5_LIBDIR=$INSTALL_DIR/lib/
    elif [[ -e /usr/include/hdf5/openmpi/H5Epublic.h ]]; then
        HDF5_INCLUDEDIR=/usr/include/hdf5/openmpi
        HDF5_LIBDIR=/usr/lib/$(uname -m)-linux-gnu/hdf5/openmpi
    elif [[ -e /usr/include/hdf5/mpich/H5Epublic.h ]]; then
        HDF5_INCLUDEDIR=/usr/include/hdf5/mpich
        HDF5_LIBDIR=/usr/lib/$(uname -m)-linux-gnu/hdf5/mpich
    fi

    #if [[ -n $SCCACHE_DIR ]]; then
    #    sccache --version
    #    echo "Using sccache"
    #    export CC="$(which sccache) mpicc"
    #    export CXX="$(which sccache) mpic++"
    #else
        export CC="mpicc"
        export CXX="mpic++"
    #fi

    CC="mpicc" \
     CXX="mpic++" \
     HDF5_MPI="ON" \
     HDF5_INCLUDEDIR=$HDF5_INCLUDEDIR \
     HDF5_LIBDIR=$HDF5_LIBDIR \
     $PIP install --no-binary=h5py h5py

    if [[ -n $SCCACHE_DIR ]]; then
        sccache --show-stats
    fi
}
