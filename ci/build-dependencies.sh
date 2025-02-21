#!/bin/bash
#set -euxo pipefail


if [[ "$GITHUB_ACTIONS" == "true" ]]; then
    export BASE=/home/runner/work/neurodamus/neurodamus
    export ENV_FILE=$GITHUB_ENV
else
    # environment variables are appended to this file, in the style of GITHUB_ENV,
    # to pass outputs between stages. I would consider this hacky
    export ENV_FILE=environment
    export BASE=/opt/ci/
fi

if [[ -e $ENV_FILE ]]; then
    echo "Old env: "
    cat $ENV_FILE

    [[ -z $GITHUB_ENV ]] && rm $ENV_FILE
fi

export VENV_PATH=$BASE/venv/
export VENV=$VENV_PATH

check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "Error: No virtual environment activated"
        exit 1
    fi
    which pip
    which pip3
}

install-apt-dependencies() {
    apt-get --yes \
        -qq \
        --no-install-suggests \
        --no-install-recommends \
        install \
        curl \
        jq \
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

install-python-dependencies() {
    check_venv

    python3 -m pip install --upgrade pip setuptools
    pip install cython numpy wheel pkgconfig mpi4py

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

    local LIBSONATAREPORT=$BASE/libsonatareport/
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

    echo "SONATAREPORT_DIR=$LIBSONATAREPORT_INSTALL" >> $ENV_FILE
}


build-neuron() {
    check_venv

    source $ENV_FILE

    local branch=$1
    local commit_id=$2

    local NRN=$BASE/nrn
    local NRN_BUILD=$NRN/build
    export NRN_INSTALL=$NRN_BUILD/install

    if [[ ! -e $NRN ]]; then
        if [[ "$branch" == 'master' ]]; then
          git clone --branch="$branch" https://github.com/neuronsimulator/nrn.git $NRN
          ( cd $NRN && git checkout $commit_id )
        else
          git clone --branch="$branch" --depth=1 https://github.com/neuronsimulator/nrn.git $NRN
        fi
    fi

    python3 -m pip install --upgrade pip -r nrn/nrn_requirements.txt

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

build-h5py() {
    check_venv


    CC="mpicc" \
        HDF5_MPI="ON" \
        HDF5_INCLUDEDIR=/usr/include/hdf5/mpich \
        HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/mpich \
        pip install --no-cache-dir --no-binary=h5py h5py --no-build-isolation
}


build-neocortex-models() {
    local branch=$1

    source $ENV_FILE

    local NEOCORTEX_MOD=$BASE/neurodamus-models/
    local NEOCORTEX_MOD_BUILD=$NEOCORTEX_MOD/build
    local NEOCORTEX_MOD_INSTALL=$NEOCORTEX_MOD_BUILD/install

    export PATH=$BASE/nrn/build/install/bin:$PATH

    if [[ ! -e $NEOCORTEX_MOD ]]; then
        git clone --branch="$branch" --depth=1 https://github.com/openbraininstitute/neurodamus-models.git $NEOCORTEX_MOD
    fi

    DATADIR=$(python3 -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

    cmake -B $NEOCORTEX_MOD_BUILD -S $NEOCORTEX_MOD  \
      -DCMAKE_INSTALL_PREFIX=$NEOCORTEX_MOD_INSTALL \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR \
      -DNEURODAMUS_CORE_DIR=${DATADIR} \
      -DNEURODAMUS_MECHANISMS=neocortex \
      -DNEURODAMUS_NCX_V5=ON

    cmake --build $NEOCORTEX_MOD_BUILD
    cmake --install $NEOCORTEX_MOD_INSTALL

    echo "NEURODAMUS_NEOCORTEX_ROOT=$NEOCORTEX_MOD_INSTALL" >> $ENV_FILE
}

build-neocortex-multiscale-models() {
    local branch=$1

    echo "build neurodamus-neocortex-multiscale model"

    source $ENV_FILE

    local NEOCORTEX_MOD_MULTI=$BASE/neurodamus-models/
    local NEOCORTEX_MOD_MULTI_BUILD=$NEOCORTEX_MOD/build
    local NEOCORTEX_MOD_MULTI_INSTALL=$NEOCORTEX_MOD_BUILD/install

    if [[ ! -e $NEOCORTEX_MOD_MULTI ]]; then
        git clone --branch="$branch" --depth=1 https://github.com/openbraininstitute/neurodamus-models.git $NEOCORTEX_MOD_MULTI
    fi

    DATADIR=$(python3 -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")


    cmake -B $NEOCORTEX_MOD_MULTI_BUILD -S $NEOCORTEX_MOD_MULTI \
      -DCMAKE_INSTALL_PREFIX=$NEOCORTEX_MOD_MULTI_INSTALL \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR \
      -DNEURODAMUS_CORE_DIR=${DATADIR} \
      -DNEURODAMUS_MECHANISMS=neocortex \
      -DNEURODAMUS_NCX_METABOLISM=ON \
      -DNEURODAMUS_NCX_NGV=ON \
      -DNEURODAMUS_ENABLE_CORENEURON=OFF

    cmake --build $NEOCORTEX_MOD_MULTI_BUILD
    cmake --install $NEOCORTEX_MOD_MULTI_INSTALL

    echo "NEURODAMUS_NEOCORTEX_MULTISCALE_ROOT=$NEOCORTEX_MOD_MULTI_INSTALL" >> $ENV_FILE
}

github-tag() {
    local repo=$1
    echo $(curl -s https://api.github.com/repos/$repo/tags | jq -r '.[0].name')
}


run-usecase() {

    source $ENV_FILE

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

    source $ENV_FILE

    #export NEURODAMUS_NEOCORTEX_ROOT=$(pwd)/neurodamus-models/install

    export PYTHONPATH=$BASE/nrn/build/install/lib/python:$PYTHONPATH
    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc
    export NEURODAMUS_PYTHON=$BASE/neurodamus/neurodamus/data
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

install() {
    apt-get update
    install-apt-dependencies

    NEURON_BRANCH=master
    NEURON_COMMIT_ID='c48d7d5'

    LIBSONATA_LATEST=`github-tag openbraininstitute/libsonata`
    LIBSONATA_REPORT_LATEST=`github-tag openbraininstitute/libsonatareport`
    NEURODAMUS_MODELS_LATEST=`github-tag openbraininstitute/neurodamus-models`

    python3 -m venv $VENV_PATH
    . $VENV_PATH/bin/activate
    export PATH    

    install-python-dependencies
    build-h5py
    build-libsonata $LIBSONATA_LATEST
    build-libsonatareport $LIBSONATA_REPORT_LATEST
    build-neuron $NEURON_BRANCH $NEURON_COMMIT_ID

    pip install neurodamus
    build-neocortex-models $NEURODAMUS_MODELS_LATEST

    pip install -r tests/requirements.txt

    run-usecase
}
