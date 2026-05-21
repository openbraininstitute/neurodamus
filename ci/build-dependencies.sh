#!/bin/bash
#set -euxo pipefail


if [[ "$GITHUB_ACTIONS" == "true" ]]; then
    export BASE=/home/runner/work/neurodamus/neurodamus
    export ENV_FILE=$GITHUB_ENV
else
    # environment variables are appended to this file, in the style of GITHUB_ENV,
    # to pass outputs between stages. I would consider this hacky
    export BASE=/opt/obi
    export ENV_FILE=$BASE/environment
fi

if [[ -e $ENV_FILE ]]; then
    echo "Old env: "
    cat $ENV_FILE

    #[[ -z $GITHUB_ENV ]] && rm $ENV_FILE
fi

CMAKE_BUILD_TYPE=RelWithDebInfo
export UV_INSTALL_DIR=$BASE/uv
export UV_CACHE_DIR=$BASE/.cache-uv
export VENV_PATH=$BASE/venv/
export INSTALL_PATH=$BASE/install
export VENV=$VENV_PATH
export PYTHON=$VENV/bin/python

export CMAKE_PREFIX_PATH=$INSTALL_PATH
#export HDF5_INCLUDE_DIRS=$INSTALL_PATH/include
#export HDF5_LIBRARIES=$INSTALL_PATH/lib
#export PKG_CONFIG_PATH=$INSTALL_PATH/lib/pkgconfig

add_to_var() {
    local var_name=$1
    local new_path=$2
    local current_value="${!var_name}"

    if [[ ":$current_value:" != *":$new_path:"* ]]; then
        export $var_name="$new_path:$current_value"
    fi
}

#add_to_var LD_LIBRARY_PATH $INSTALL_PATH/lib64
#add_to_var LD_LIBRARY_PATH $INSTALL_PATH/lib

function install-uv() {
    if [[ ! -f $UV_INSTALL_DIR/uv ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$UV_INSTALL_DIR sh
    fi
}

[[ -f $UV_INSTALL_DIR/env ]] && source $UV_INSTALL_DIR/env

function make-venv() {
    echo "Create venv and install some basic packages"
    if [[ ! -d $VENV ]]; then
        #uv venv --python 3.12 --no-project $VENV_PATH
        python3.11 -mvenv $VENV_PATH
    fi
}

[[ -f $VENV_PATH/bin/activate ]] && source $VENV_PATH/bin/activate
export PIP=$VENV_PATH/bin/pip
#export PIP='uv pip'

install-apt-dependencies() {
    apt-get --yes \
        -qq \
        --no-install-suggests \
        --no-install-recommends \
        install \
        curl \
        ca-certificates \
        jq \
        make \
        g++ \
        gcc \
        python3 \
        python3-dev \
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

install-hdf5() {
    local branch=hdf5_1.14.6
    local HDF5=$BASE/hdf5
    local HDF5_BUILD=$HDF5/build

    git clone --branch="$branch" --shallow-submodules --depth=1 https://github.com/HDFGroup/hdf5/ $HDF5

    module load openmpi5/5.0.9amzn1 libfabric-aws/2.4.0amzn1.0
    # note that `HDF5_USE_FILE_LOCKING` is off, this saves loading time of the
    # read only data.  Since we don't use `SWMR` writing of files, this should
    # be safe.
    # see: https://support.hdfgroup.org/documentation/hdf5/latest/_file_lock.html
    cmake -B "$HDF5_BUILD" -GNinja \
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
        -DCMAKE_C_COMPILER=`which mpicc` \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
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
        -DHDF5_USE_FILE_LOCKING=OFF \
        -S "$HDF5"

    cmake --build "$HDF5_BUILD" -j
    cmake --install "$HDF5_BUILD"
}

install-dnf-dependencies() {
    dnf -y install \
        bison \
        cpp \
        cmake \
        gcc-c++ \
        flex \
        git \
        libfl-devel \
        readline-devel \
        python3.11-devel python3.11-pip python3-virtualenv \
        ninja-build
}

install-rpm-dependencies() {
    yum -y install \
        bison \
        cpp \
        cmake \
        gcc-c++ \
        flex \
        git \
        flex-devel \
        readline-devel \
        ninja-build
        #python3.11-devel python3.11-pip python3-virtualenv \
}

pre() {
    module purge
    module load openmpi5 libfabric-aws
    module list
    hash -l
    which mpicc
}

install-python-dependencies() {
    pre

    $PIP install --upgrade pip setuptools
    $PIP install cython numpy wheel pkgconfig jinja2 pyyaml cmake
    MPICC=mpicc $PIP install --no-binary=mpi4py mpi4py
}

build-libsonata() {
    pre

    local branch=${1:-master}
    local LIBSONATA=$BASE/libsonata/

    git clone --branch="$branch" --shallow-submodules --depth=1 https://github.com/openbraininstitute/libsonata $LIBSONATA
    (cd $LIBSONATA && \
        git submodule update --init --depth=1 \
        extlib/HighFive \
        extlib/fmt \
        python/pybind11
    )

    SONATA_BUILD_TYPE=$CMAKE_BUILD_TYPE CC=mpicc CXX=mpic++ $PIP -v install $LIBSONATA
}

build-libsonatareport() {
    local default=`github-tag openbraininstitute/libsonatareport`
    local branch=${1:-$default}

    local LIBSONATAREPORT=$BASE/libsonatareport/
    local LIBSONATAREPORT_BUILD=$LIBSONATAREPORT/build

    git clone --branch="$branch" --depth=1 \
        https://github.com/openbraininstitute/libsonatareport.git \
        $LIBSONATAREPORT

    (cd $LIBSONATAREPORT && \
        git submodule update --init --depth=1 \
        extlib/spdlog \
        extlib/Catch2 \
    )

    module purge
    module load openmpi5/5.0.9amzn1 libfabric-aws/2.4.0amzn1.0

    cmake \
        -B $LIBSONATAREPORT_BUILD \
        -S $LIBSONATAREPORT \
        -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
        -DSONATA_REPORT_ENABLE_SUBMODULES=ON \
        -DSONATA_REPORT_ENABLE_MPI=ON \
        -DSONATA_REPORT_ENABLE_TEST=OFF \
        ..

    cmake --build $LIBSONATAREPORT_BUILD --parallel
    cmake --build $LIBSONATAREPORT_BUILD --target install

    echo "SONATAREPORT_DIR=$INSTALL_PATH" >> $ENV_FILE
}


build-neuron() {
    pre
    source $ENV_FILE

    local branch=$1
    local commit_id=$2

    local NRN=$BASE/nrn
    local NRN_BUILD=$NRN/build

    if [[ ! -e $NRN ]]; then
        if [[ "$branch" == 'master' ]]; then
          git clone --branch="$branch" https://github.com/neuronsimulator/nrn.git $NRN
          ( cd $NRN && git checkout $commit_id )
        else
          git clone --branch="$branch" --depth=1 https://github.com/neuronsimulator/nrn.git $NRN
        fi
    fi

    (cd $NRN && \
        git submodule update --init --depth=1 --recursive \
        external/spdlog \
        external/Random123 \
        external/fmt \
        external/nanobind \
    )

    #$PIP install --upgrade pip -r nrn/nrn_requirements.txt

    cmake -B $NRN_BUILD -S $NRN \
      -G Ninja \
      -DPYTHON_EXECUTABLE=$VENV/bin/python \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
      -DNRN_ENABLE_MPI=ON \
      -DNRN_ENABLE_RX3D=OFF \
      -DNRN_ENABLE_INTERVIEWS=OFF \
      -DNRN_ENABLE_CORENEURON=ON \
      -DNMODL_ENABLE_PYTHON_BINDINGS=OFF \
      -DCORENRN_ENABLE_REPORTING=ON \
      -DCMAKE_PREFIX_PATH=$INSTALL_PATH

    cmake --build $NRN_BUILD --parallel
    cmake --build $NRN_BUILD --target install

    ( cd $VENV_PATH/lib/python*/site-packages && ln -s $INSTALL_PATH/lib/python neuron )

    echo "NRN_INSTALL=$INSTALL_PATH" >> $ENV_FILE
    echo "PATH=$INSTALL_PATH/bin:$PATH" >> $ENV_FILE
}

build-h5py() {
    if [ -d /opt/obi/install ]; then
        HDF5_INCLUDEDIR=/opt/obi/install/include
        HDF5_LIBDIR=/opt/obi/install/lib
    #else
    #    HDF5_INCLUDEDIR=/usr/include/hdf5/mpich
    #    HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/mpich
    fi

    CC="mpicc" \
        HDF5_MPI="ON" \
        HDF5_INCLUDEDIR=$HDF5_INCLUDEDIR \
        HDF5_LIBDIR=$HDF5_LIBDIR \
        $PIP -v install --no-cache-dir --no-binary=h5py h5py --no-build-isolation
}

install-neurodamus() {
    git clone --depth 1 https://github.com/openbraininstitute/neurodamus/
    $PIP -v install -e neurodamus
}


build-neocortex-models() {
    #local default=`github-tag openbraininstitute/neurodamus-models`
    local default=main
    local branch=${1:-$default}

    source $ENV_FILE

    local NEOCORTEX_MOD=$BASE/neurodamus-models/
    local NEOCORTEX_MOD_BUILD=$NEOCORTEX_MOD/build
    local NEOCORTEX_MOD_INSTALL=$NEOCORTEX_MOD/build/install

    export PATH=$INSTALL_PATH/bin:$PATH

    if [[ ! -e $NEOCORTEX_MOD ]]; then
        git clone --branch="$branch" --depth=1 https://github.com/openbraininstitute/neurodamus-models.git $NEOCORTEX_MOD
    fi

    DATADIR=$($PYTHON -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

    echo "Datadir is: $DATADIR"

    cmake -B $NEOCORTEX_MOD_BUILD -S $NEOCORTEX_MOD  \
      -DCMAKE_INSTALL_PREFIX=$NEOCORTEX_MOD_INSTALL \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR \
      -DNEURODAMUS_CORE_DIR=${DATADIR} \
      -DNEURODAMUS_MECHANISMS=neocortex \
      -DNEURODAMUS_NCX_V5=ON

    cmake --build $NEOCORTEX_MOD_BUILD
    cmake --install $NEOCORTEX_MOD_BUILD

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

    DATADIR=$(PYTHON -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")


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
    cmake --install $NEOCORTEX_MOD_MULTI_BUILD

    echo "NEURODAMUS_NEOCORTEX_MULTISCALE_ROOT=$NEOCORTEX_MOD_MULTI_INSTALL" >> $ENV_FILE
}

github-tag() {
    local repo=$1
    echo $(curl -s https://api.github.com/repos/$repo/tags | jq -r '.[0].name')
}


run-usecase() {
    module purge
    module load openmpi5 libfabric-aws
    which mpirun

    source $ENV_FILE

    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc

    export NEURODAMUS_PYTHON=$BASE/neurodamus/neurodamus/data
    #export CORENEURONLIB=$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so
    export NRNMECH_LIB_PATH=$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so
    export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH

    #echo "PYTHONPATH=$PYTHONPATH" >> $GITHUB_ENV;
    #echo "HOC_LIBRARY_PATH=$HOC_LIBRARY_PATH" >> $GITHUB_ENV;
    #echo "CORENEURONLIB=$CORENEURONLIB" >> $GITHUB_ENV;
    #echo "NRNMECH_LIB_PATH=$NRNMECH_LIB_PATH" >> $GITHUB_ENV;

    which special

    pushd $BASE/neurodamus/
    # launch simulation with NEURON
    pushd tests/simulations/usecase3/
    mpirun --allow-run-as-root --use-hwthread-cpus -np 2 special -mpi -python $NEURODAMUS_PYTHON/init.py --configFile=simulation_sonata.json --output-path=`pwd`
    ls reporting/*.h5

    ## launch simulation with CORENEURON
    #mpirun --allow-run-as-root --use-hwthread-cpus -np 2 special -mpi -python $NEURODAMUS_PYTHON/init.py --configFile=simulation_sonata_coreneuron.json
    #ls reporting_coreneuron/*.h5
    #popd
}

run-pytest() {

    set -a
    source $ENV_FILE
    set +a

    export NEURODAMUS_PYTHON=/neurodamus/neurodamus/data
    export PYTHONPATH=$NRN_INSTALL/lib/python:$PYTHONPATH
    export RDMAV_FORK_SAFE=1
    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc
    export CORENEURONLIB=$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so
    export NRNMECH_LIB_PATH=$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so
    export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH

    #pytest -v -s -x --durations=5 --durations-min=15  'tests/integration-e2e/test_reports.py::test_reports_cell_permute[create_tmp_simulation_config_file0]'
    pytest --monkeytype-output=/opt/obi/neurodamus/monkeytype.sqlite3 -v -s -x --durations=5 --durations-min=15 --forked tests/integration-e2e/
}

run-pytest-sci() {
    set -a
    source $ENV_FILE
    set +a

    export RDMAV_FORK_SAFE=1

    export NEURODAMUS_PYTHON=/neurodamus/neurodamus/data
    export PYTHONPATH=$NRN_INSTALL/lib/python:$PYTHONPATH
    export NEURODAMUS_NEOCORTEX_ROOT=/opt/obi/neurodamus-models//build/install
    export HOC_LIBRARY_PATH=$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc
    export CORENEURONLIB=$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so
    export NRNMECH_LIB_PATH=$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so
    export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH

    pytest --monkeytype-output=/opt/obi/neurodamus/monkeytype.sqlite3 -x -s --forked --durations=5 --durations-min=15 tests/scientific/
    #pytest -x -s --forked --durations=5 --durations-min=15 tests/scientific/test_multipop.py
    #pytest -x -s --durations=5 --durations-min=15 tests/scientific/test_lfp.py::test_v5_sonata_lfp
}

install() {
    apt-get update
    install-apt-dependencies

    NEURON_BRANCH=master
    NEURON_COMMIT_ID='0d990513b'

    #install-uv
    make-venv
    install-python-dependencies
    install-hdf5
    build-h5py
    build-libsonata `github-tag openbraininstitute/libsonata`
    build-libsonatareport `github-tag openbraininstitute/libsonatareport`
    build-neuron $NEURON_BRANCH $NEURON_COMMIT_ID
    #build-neuron master 0d990513b

    # XXX need to install neurodamus before this!
    build-neocortex-models `github-tag openbraininstitute/neurodamus-models`

    #$PIP install -r tests/requirements.txt

    run-usecase
}
