#!/bin/bash

install-h5py() {
   set -x
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

    if [[ -n $SCCACHE_DIR ]]; then
        echo "Using sccache"
        export CC="sccache mpicc"
        export CXX="sccache mpic++"
    else
        export CC="mpicc"
        export CXX="mpic++"
    fi

    HDF5_MPI="ON" \
    HDF5_INCLUDEDIR=$HDF5_INCLUDEDIR \
    HDF5_LIBDIR=$HDF5_LIBDIR \
    $PIP -v install --no-cache-dir --no-binary=h5py h5py --no-build-isolation
}
