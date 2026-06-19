ARG UV_VERSION=0.11.15
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM python:${PYTHON_VERSION}-slim

ENV SCCACHE_DIR=/var/cache/sccache
# Docker Args are: LIBSONATAREPORT_COMMIT LIBSONATA_COMMIT NEURON_COMMIT NEURODAMUS_COMMIT
# they are defined at the point of usage, to reduce the layers that need to be rebuilt

ENV USER_VENV=/workspace/user_venv

ENV INSTALL_DIR=/opt/obi
ENV BUILD_DIR=/tmp

ENV CMAKE_BUILD_TYPE=RelWithDebugInfo

COPY --from=uv /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_=PYTHON_DOWNLOADSnever \
    UV_PYTHON=python${PYTHON_VERSION}

SHELL ["/bin/bash", "-c"]
WORKDIR /workspace

RUN --mount=type=bind,source=ci/scripts/install-apt-dependencies.sh,target=/tmp/install-apt-dependencies.sh \
    apt-get --yes -qq update \
    && apt-get --yes -qq upgrade \
    && source /tmp/install-apt-dependencies.sh \
    && install-apt-dependencies \
    && apt-get --yes -qq --no-install-recommends install libopenmpi-dev libhdf5-openmpi-dev \
    && apt-get --yes -qq clean \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv $USER_VENV

RUN --mount=type=bind,source=ci/scripts/install-python-dependencies.sh,target=/tmp/install-python-dependencies.sh \
    --mount=type=cache,target=/root/.cache/uv \
    source /tmp/install-python-dependencies.sh \
    && source $USER_VENV/bin/activate \
    && PIP='uv pip' install-python-dependencies

RUN --mount=type=bind,source=ci/scripts/install-sccache.sh,target=/tmp/install-sccache.sh \
    source /tmp/install-sccache.sh \
	&& install-sccache

RUN --mount=type=bind,source=ci/scripts/install-h5py.sh,target=/tmp/install-h5py.sh \
    mkdir -p /tmp/stable-build \
    && source /tmp/install-h5py.sh \
    && source $USER_VENV/bin/activate \
    && PIP='uv pip' install-h5py

ARG LIBSONATAREPORT_COMMIT=2.0.0

RUN --mount=type=bind,source=ci/scripts/build-libsonatareport.sh,target=/tmp/build-libsonatareport.sh \
    --mount=type=cache,target=/var/cache/sccache \
    source /tmp/build-libsonatareport.sh \
    && source $USER_VENV/bin/activate \
    && build-libsonatareport $LIBSONATAREPORT_COMMIT

ARG LIBSONATA_COMMIT=v0.1.35

RUN --mount=type=bind,source=ci/scripts/build-libsonata.sh,target=/tmp/build-libsonata.sh \
    --mount=type=cache,target=/root/.cache/uv \
    source /tmp/build-libsonata.sh \
    && source $USER_VENV/bin/activate \
    && PIP='uv pip' build-libsonata $LIBSONATA_COMMIT

ARG NEURON_COMMIT=9.0.1

RUN --mount=type=bind,source=ci/scripts/build-neuron.sh,target=/tmp/build-neuron.sh \
    --mount=type=cache,target=/var/cache/sccache \
    --mount=type=cache,target=/root/.cache/uv \
    source /tmp/build-neuron.sh \
    && source $USER_VENV/bin/activate \
    && PIP='uv pip' build-neuron $NEURON_COMMIT

ARG NEURODAMUS_COMMIT=c46a58f9bcf72522732fe0dbc2c1456484a8c737

RUN --mount=type=bind,source=ci/scripts/build-neurodamus.sh,target=/tmp/build-neurodamus.sh \
    --mount=type=cache,target=/root/.cache/uv \
    source /tmp/build-neurodamus.sh \
    && source $USER_VENV/bin/activate \
    && PIP='uv pip' build-neurodamus $NEURODAMUS_COMMIT

RUN --mount=type=bind,source=ci/scripts/make-neurodamus-nrnivmodl.sh,target=/tmp/make-neurodamus-nrnivmodl.sh \
    source $USER_VENV/bin/activate \
    && source /tmp/make-neurodamus-nrnivmodl.sh \
    && make-neurodamus-nrnivmodl

RUN --mount=type=bind,source=ci/scripts/make-neocortex-env.sh,target=/tmp/make-neocortex-env.sh \
    source $USER_VENV/bin/activate \
    && export PATH=/opt/obi:$PATH \
    && source /tmp/make-neocortex-env.sh \
    && make-neocortex-env
