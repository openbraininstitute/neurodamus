# export CMAKE_BUILD_TYPE=RelWithDebugInfo
# export BUILD_DIR=/tmp
# export INSTALL_DIR=/opt/obi

install-hdf5() {
    local branch=hdf5_1.14.6
    local HDF5=$BUILD_DIR/hdf5
    local HDF5_BUILD=$HDF5/build

    git clone --branch="$branch" --shallow-submodules --depth=1 https://github.com/HDFGroup/hdf5/ $HDF5

    cmake -B "$HDF5_BUILD" -GNinja \
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
        -DCMAKE_C_COMPILER=`which mpicc` \
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
        -S "$HDF5"

    cmake --build "$HDF5_BUILD" --parallel
    cmake --install "$HDF5_BUILD"
}
