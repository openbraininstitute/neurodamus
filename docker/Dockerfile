FROM ubuntu:22.04
LABEL org.opencontainers.image.authors="Weina Ji <weina.ji@openbraininstitute.org>"

# The default shell for the RUN instruction is ["/bin/sh", "-c"].
# Using SHELL instruction to change default shell for subsequent RUN instructions
SHELL ["/bin/bash", "-c"]

ARG LIBSONATA_TAG=master
ARG LIBSONATAREPORT_TAG=master
ARG NEURON_TAG=master
ARG NEURON_COMMIT_ID
ARG WORKDIR=/opt/software
ARG INSTALL_DIR=/opt/software/install
ARG USR_VENV=$WORKDIR/venv

# Install needed libs
RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && apt-get --yes -qq install \
                      g++ \
                      gcc \
                      python3.10 \
                      python3-pip \
                      python3-venv \
                      git \
                      cmake \
                      wget \
                      vim \
                      mpich libmpich-dev libhdf5-mpich-dev hdf5-tools \
                      flex libfl-dev bison ninja-build libreadline-dev \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv $USR_VENV \
 && source $USR_VENV/bin/activate \
 && pip install -U pip setuptools \
 && pip install -U cython pytest sympy jinja2 pyyaml numpy \
                   numpy wheel pkgconfig

# Install libsonata
RUN source $USR_VENV/bin/activate \
 &&CC=mpicc CXX=mpic++ pip install git+https://github.com/openbraininstitute/libsonata@$LIBSONATA_TAG

# Install libsonatareport
RUN mkdir -p $WORKDIR \
 && cd $WORKDIR \
 && git clone https://github.com/openbraininstitute/libsonatareport.git --recursive --depth 1 -b $LIBSONATAREPORT_TAG \
 && cmake -B rep_build -S libsonatareport -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DSONATA_REPORT_ENABLE_SUBMODULES=ON -DSONATA_REPORT_ENABLE_MPI=ON .. \
 && cmake --build rep_build --parallel \
 && cmake --install rep_build \
 && rm -rf libsonatareport rep_build
ENV SONATAREPORT_DIR="$INSTALL_DIR"

# Install neuron
RUN source $USR_VENV/bin/activate \ 
 && cd $WORKDIR \
 && if [[ ! -z $NEURON_COMMIT_ID ]]; then \
    git clone https://github.com/neuronsimulator/nrn.git ;\
    cd nrn ; \
    git checkout $NEURON_COMMIT_ID ; \
    cd .. ; \
    else git clone https://github.com/neuronsimulator/nrn.git --depth 1 -b $NEURON_TAG ; \
    fi \
 && cmake -B nrn_build -S nrn -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_RX3D=OFF \
 -DNRN_ENABLE_CORENEURON=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCORENRN_ENABLE_REPORTING=ON -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR \
 && cmake --build nrn_build -- -j 2 \
 && cmake --install nrn_build \
 && rm -rf nrn nrn_build

# Build h5py with the local hdf5
RUN source $USR_VENV/bin/activate \
 && pip install mpi4py \
 && ARCH=$(uname -m) CC="mpicc" HDF5_MPI="ON" HDF5_INCLUDEDIR=/usr/include/hdf5/mpich HDF5_LIBDIR=/usr/lib/$ARCH-linux-gnu/hdf5/mpich \
    pip install --no-cache-dir --no-binary=h5py h5py --no-build-isolation

# Install neurodamus and prepare HOC_LIBRARY_PATH
RUN source $USR_VENV/bin/activate \
 && cd $WORKDIR \
 && git clone https://github.com/openbraininstitute/neurodamus.git \
 && cd neurodamus \
 && pip install .

ENV HOC_LIBRARY_PATH="$WORKDIR/neurodamus/neurodamus/data/hoc"
ENV NEURODAMUS_PYTHON="$WORKDIR/neurodamus/neurodamus/data"
ENV NEURODAMUS_MODS_DIR="$WORKDIR/neurodamus/neurodamus/data/mod"
ENV PATH="$INSTALL_DIR/bin:$USR_VENV/bin:$PATH"
ENV PYTHONPATH="$INSTALL_DIR/lib/python:$PYTHONPATH"

# Copy common bluebrain hoc and mod files from neurodamus-models, required for instantiating neurodamus
RUN wget -q https://raw.githubusercontent.com/openbraininstitute/neurodamus-models/refs/heads/main/common/hoc/AMPANMDAHelper.hoc -O $HOC_LIBRARY_PATH/AMPANMDAHelper.hoc \
 && wget -q https://raw.githubusercontent.com/openbraininstitute/neurodamus-models/refs/heads/main/common/hoc/GABAABHelper.hoc -O $HOC_LIBRARY_PATH/GABAABHelper.hoc \
 && wget -q https://raw.githubusercontent.com/openbraininstitute/neurodamus-models/refs/heads/main/common/mod/ProbAMPANMDA_EMS.mod -O $NEURODAMUS_MODS_DIR/ProbAMPANMDA_EMS.mod \
 && wget -q https://raw.githubusercontent.com/openbraininstitute/neurodamus-models/refs/heads/main/common/mod/ProbGABAAB_EMS.mod -O $NEURODAMUS_MODS_DIR/ProbGABAAB_EMS.mod

# Edit module building script and test build
ADD build_neurodamus.sh $INSTALL_DIR/bin
RUN ARCH=$(uname -m) \
 && sed -i "s/ARCH=\"x86_64\"/ARCH=\"$ARCH\"/g" $INSTALL_DIR/bin/build_neurodamus.sh \
 && build_neurodamus.sh $NEURODAMUS_MODS_DIR \
 && ./$ARCH/special -python -c "from neuron import h; h.quit()" \
 && ./$ARCH/special -python -c "from neurodamus.core import NeuronWrapper as Nd; Nd.init()" \
 && rm -rf $ARCH/

ENTRYPOINT ["bash"]
