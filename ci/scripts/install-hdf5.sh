#!/usr/bin/env bash
# Build and install HDF5 with parallel (MPI) support from source.
#
# Environment variables:
#   PRE              called on function entry
#   BUILD_DIR        Directory for cloning/building source (e.g. /tmp)
#   INSTALL_DIR      Prefix where HDF5 will be installed (e.g. /opt/obi)
#   CMAKE_BUILD_TYPE CMake build type (e.g. RelWithDebugInfo)

install-hdf5() {
    PRE || true

    local branch=hdf5_1.14.6
    local HDF5=$BUILD_DIR/hdf5
    local HDF5_BUILD=$HDF5/build

    [[ -e $HDF5 ]] || \
      git clone --branch="$branch" --shallow-submodules --depth=1 https://github.com/HDFGroup/hdf5/ $HDF5

    # have to use `Release`; RelWithDebugInfo causes `h5pcc` wrapper to pass -lhdf5_debug eventhough
    # it is not included
    local CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=`type mpicc` \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DHDF5_ENABLE_PARALLEL=ON \
        -DHDF5_ENABLE_NONSTANDARD_FEATURES=OFF \
        -DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16=OFF \
        -DHDF5_BUILD_STATIC_TOOLS=OFF \
        -DHDF5_BUILD_UTILS=OFF \
        -DHDF5_BUILD_HL_LIB=ON \
        -DHDF5_BUILD_EXAMPLES=OFF \
        -DBUILD_STATIC_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_ENABLE_SZIP_ENCODING=OFF \
        -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
        -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF \
    )

    if [[ -n $SCCACHE_DIR ]]; then
        CMAKE_ARGS+=(
            -DCMAKE_C_COMPILER_LAUNCHER=sccache
            -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
        )
    fi


    cmake -B "$HDF5_BUILD" -GNinja \
        -S "$HDF5" \
        "${CMAKE_ARGS[@]}"

    cmake --build "$HDF5_BUILD" --parallel
    cmake --install "$HDF5_BUILD"

    if [[ -n $SCCACHE_DIR ]]; then
        sccache --show-stats
    fi
}
