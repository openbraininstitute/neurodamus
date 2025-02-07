#!/bin/bash
#set -euxo pipefail

VENV=/venv

check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "Error: No virtual environment activated"
        exit 1
    fi
}

install-apt-dependencies() {
    apt-get --yes \
        -qq \
        --no-install-suggests \
        --no-install-recommends \
        install \
        make \
        g++ \
        gcc \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        cmake \
        wget \
        vim \
        mpich \
        libmpich-dev \
        libhdf5-mpich-dev \
        hdf5-tools \
        flex \
        libfl-dev \
        bison \
        ninja-build \
        libreadline-dev
}

build-libsonata() {
    check_venv

    local branch=$1

    git clone --branch="$branch" --shallow-submodules --depth=1 https://github.com/openbraininstitute/libsonata
    (cd libsonata && \
        git submodule update --init --depth=1 \
        extlib/HighFive \
        extlib/fmt \
        python/pybind11
    )

    CC=mpicc CXX=mpic++ pip install --no-cache-dir --no-binary=libsonata libsonata/
}

build-libsonatareport() {
    local branch=$1

    local LIBSONATAREPORT=$(pwd)/libsonatareport/
    local LIBSONATAREPORT_BUILD=$LIBSONATAREPORT/build
    local LIBSONATAREPORT_INSTALL=$LIBSONATAREPORT_BUILD/install

    git clone --branch="$branch" --depth=1 \
        https://github.com/openbraininstitute/libsonatareport.git \
        $LIBSONATAREPORT
    (cd $LIBSONATAREPORT && \
        git submodule update --init --depth=1 \
        extlib/spdlog \
        extlib/Catch2
    )

    cmake \
        -B $LIBSONATAREPORT_BUILD \
        -S $LIBSONATAREPORT \
        -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$LIBSONATAREPORT_INSTALL" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSONATA_REPORT_ENABLE_SUBMODULES=ON \
        -DSONATA_REPORT_ENABLE_MPI=ON \
        -DSONATA_REPORT_ENABLE_TEST=OFF \
        ..

    cmake --build $LIBSONATAREPORT_BUILD --parallel
    cmake --build $LIBSONATAREPORT_BUILD --target install
}


build-neuron() {
    check_venv

    local branch=$1
    local commit_id=$2

    local SONATAREPORT_DIR=$(pwd)/libsonatareport/build/install

    #XXX
    export SONATAREPORT_DIR=/opt/ci/libsonatareport/build/install/

    local NRN=$(pwd)/nrn
    local NRN_BUILD=$NRN/build
    local NRN_INSTALL=$NRN_BUILD/install

    if [[ ! -e $NRN ]]; then
        if [[ "$branch" == 'master' ]]; then
          git clone --branch="$branch" https://github.com/neuronsimulator/nrn.git $NRN
          ( cd $NRN && git checkout $commit_id )
        else
          git clone --branch="$branch" --depth=1 https://github.com/neuronsimulator/nrn.git $NRN
        fi
    fi

    python -m pip install --upgrade pip -r nrn/nrn_requirements.txt

    cmake -B $NRN_BUILD -S $NRN \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DCMAKE_INSTALL_PREFIX=$NRN_INSTALL \
      -DNRN_ENABLE_MPI=ON \
      -DNRN_ENABLE_INTERVIEWS=OFF \
      -DNRN_ENABLE_CORENEURON=ON \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCORENRN_ENABLE_REPORTING=ON \
      -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR
    cmake --build nrn/build --parallel
    cmake --build nrn/build --target install
}

build-neocortex-models() {
    local branch=$1

    export SONATAREPORT_DIR=$(pwd)/libsonatareport/build/install
    #XXX
    export SONATAREPORT_DIR=/opt/ci/libsonatareport/build/install/

    export PATH=$(pwd)/nrn/build/install/bin:$PATH

    # Clone neurodamus-models repository
    git clone --branch="$branch" --depth=1 https://github.com/openbraininstitute/neurodamus-models.git

    # Build neocortex model
    DATADIR=$(python -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

    cmake -B neurodamus-models/build -S neurodamus-models/ \
      -DCMAKE_INSTALL_PREFIX=$PWD/neurodamus-models/install \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR \
      -DNEURODAMUS_CORE_DIR=${DATADIR} \
      -DNEURODAMUS_MECHANISMS=neocortex \
      -DNEURODAMUS_NCX_V5=ON

    cmake --build neurodamus-models/build
    cmake --install neurodamus-models/build
    echo "NEURODAMUS_NEOCORTEX_ROOT=$(pwd)/neurodamus-models/install" #>> $GITHUB_ENV;
}

build-h5py() {
    check_venv

CC="mpicc" HDF5_MPI="ON" HDF5_INCLUDEDIR=/usr/include/hdf5/mpich HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/mpich   pip install --no-cache-dir --no-binary=h5py h5py --no-build-isolation
}

run-usecase() {

    export NEURODAMUS_NEOCORTEX_ROOT=$(pwd)/neurodamus-models/install

    export PYTHONPATH=$(pwd)/nrn/build/install/lib/python:$PYTHONPATH
    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc
    export NEURODAMUS_PYTHON=$(pwd)/neurodamus/neurodamus/data
    export CORENEURONLIB=$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so
    export NRNMECH_LIB_PATH=$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so
    export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH

    #echo "PYTHONPATH=$PYTHONPATH" >> $GITHUB_ENV;
    #echo "HOC_LIBRARY_PATH=$HOC_LIBRARY_PATH" >> $GITHUB_ENV;
    #echo "CORENEURONLIB=$CORENEURONLIB" >> $GITHUB_ENV;
    #echo "NRNMECH_LIB_PATH=$NRNMECH_LIB_PATH" >> $GITHUB_ENV;
    #echo "NEURODAMUS_PYTHON=$NEURODAMUS_PYTHON" >> $GITHUB_ENV;

    which special

    # launch simulation with NEURON
    cd tests/simulations/usecase3/
    mpirun -np 2 special -mpi -python $NEURODAMUS_PYTHON/init.py --configFile=simulation_sonata.json
    ls reporting/*.h5
    # launch simulation with CORENEURON
    mpirun -np 2 special -mpi -python $NEURODAMUS_PYTHON/init.py --configFile=simulation_sonata_coreneuron.json
    ls reporting_coreneuron/*.h5
}

run-pytest() {

    export NEURODAMUS_NEOCORTEX_ROOT=$(pwd)/neurodamus-models/install

    export PYTHONPATH=$(pwd)/nrn/build/install/lib/python:$PYTHONPATH
    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc
    export NEURODAMUS_PYTHON=$(pwd)/neurodamus/neurodamus/data
    export CORENEURONLIB=$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so
    export NRNMECH_LIB_PATH=$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so
    export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH

    pytest -s -x --forked --durations=5 --durations-min=15 tests/integration-e2e
}

run-pytest-sci() {

    export NEURODAMUS_NEOCORTEX_ROOT=$(pwd)/neurodamus-models/install

    export PYTHONPATH=$(pwd)/nrn/build/install/lib/python:$PYTHONPATH
    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc
    export NEURODAMUS_PYTHON=$(pwd)/neurodamus/neurodamus/data
    export CORENEURONLIB=$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so
    export NRNMECH_LIB_PATH=$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so
    export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH

    pytest -s -x --forked --durations=5 --durations-min=15 tests/scientific
}
