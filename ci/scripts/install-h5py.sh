#!/usr/bin/env bash
# Install h5py with MPI support, building from source against the available HDF5 library.
#
# Environment variables:
#   PRE         called on function entry
#   INSTALL_DIR Prefix where HDF5 may be installed (checked first)
#   PIP         pip command to use (e.g. "uv pip")

install-h5py() {
    PRE || true

    if [[ -e $INSTALL_DIR/include/H5Epublic.h ]]; then
        HDF5_INCLUDEDIR=$INSTALL_DIR/include/
        HDF5_LIBDIR=$INSTALL_DIR/lib/
    elif [[ -e /usr/include/hdf5/openmpi/H5Epublic.h ]]; then
        HDF5_INCLUDEDIR=/usr/include/hdf5/openmpi
        HDF5_LIBDIR=/usr/lib/$(uname -m)-linux-gnu/hdf5/openmpi
    elif [[ -e /usr/include/hdf5/mpich/H5Epublic.h ]]; then
        HDF5_INCLUDEDIR=/usr/include/hdf5/mpich
        HDF5_LIBDIR=/usr/lib/$(uname -m)-linux-gnu/hdf5/mpich
    elif [[ $(uname) == Darwin ]]; then
        HDF5_INCLUDEDIR=$(brew --prefix hdf5-mpi)/include
        HDF5_LIBDIR=$(brew --prefix hdf5-mpi)/lib
    fi

    CC="mpicc" \
     CXX="mpic++" \
     HDF5_MPI="ON" \
     HDF5_INCLUDEDIR=$HDF5_INCLUDEDIR \
     HDF5_LIBDIR=$HDF5_LIBDIR \
     $PIP install -v --no-binary=h5py h5py
}
