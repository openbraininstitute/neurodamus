ARG UV_VERSION=0.11.15
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM python:${PYTHON_VERSION}-slim

ENV SCCACHE_DIR=/var/cache/sccache

ARG LIBSONATAREPORT_COMMIT=de0abd0e73f29975ec7caeb80118bd617f2dbe0c
ARG LIBSONATA_COMMIT=v0.1.35
ARG NEURODAMUS_COMMIT=4.2.1
ARG NEURODAMUS_MODELS_COMMIT=1a3b8f98fbabefb34f255c1fe63f7d7a4422223f
ARG NEURON_COMMIT=9.0.1

ENV USER_VENV_NAME=user_venv

ENV INSTALL_DIR=/opt/obi
ENV BUILD_DIR=/tmp

ENV CMAKE_BUILD_TYPE=RelWithDebugInfo

COPY --from=uv /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python${PYTHON_VERSION}

SHELL ["/bin/bash", "-c"]
WORKDIR /workspace

RUN --mount=type=bind,source=ci/scripts/install-apt-dependencies.sh,target=/tmp/install-apt-dependencies.sh \
    apt-get --yes -qq update \
    && apt-get --yes -qq upgrade \
    && source /tmp/install-apt-dependencies.sh \
    && install-apt-dependencies \
    && apt-get --yes -qq --no-install-recommends install libhdf5-openmpi-dev \
    && apt-get --yes -qq clean \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /workspace/user_venv

RUN --mount=type=bind,source=ci/scripts/install-python-dependencies.sh,target=/tmp/install-python-dependencies.sh \
    --mount=type=cache,target=/root/.cache/uv \
    source /tmp/install-python-dependencies.sh \
    && source /workspace/user_venv/bin/activate \
    && PIP='uv pip' install-python-dependencies

RUN --mount=type=bind,source=ci/scripts/install-sccache.sh,target=/tmp/install-sccache.sh \
    source /tmp/install-sccache.sh \
	&& install-sccache

RUN --mount=type=bind,source=ci/scripts/install-h5py.sh,target=/tmp/install-h5py.sh \
    mkdir -p /tmp/stable-build \
    && source /tmp/install-h5py.sh \
    && source /workspace/user_venv/bin/activate \
    && PIP='uv pip' install-h5py

RUN --mount=type=bind,source=ci/scripts/build-libsonatareport.sh,target=/tmp/build-libsonatareport.sh \
    --mount=type=cache,target=/var/cache/sccache \
    source /tmp/build-libsonatareport.sh \
    && source /workspace/user_venv/bin/activate \
    && build-libsonatareport $LIBSONATAREPORT_COMMIT

RUN --mount=type=bind,source=ci/scripts/build-libsonata.sh,target=/tmp/build-libsonata.sh \
    --mount=type=cache,target=/root/.cache/uv \
    source /tmp/build-libsonata.sh \
    && source /workspace/user_venv/bin/activate \
    && PIP='uv pip' build-libsonata $LIBSONATA_COMMIT

RUN --mount=type=bind,source=ci/scripts/build-neuron.sh,target=/tmp/build-neuron.sh \
    --mount=type=cache,target=/var/cache/sccache \
    --mount=type=cache,target=/root/.cache/uv \
    source /tmp/build-neuron.sh \
    && source /workspace/user_venv/bin/activate \
    && PIP='uv pip' build-neuron $NEURON_COMMIT

RUN --mount=type=bind,source=ci/scripts/build-neurodamus.sh,target=/tmp/build-neurodamus.sh \
    --mount=type=cache,target=/root/.cache/uv \
    source /tmp/build-neurodamus.sh \
    && source /workspace/user_venv/bin/activate \
    && PIP='uv pip' build-neurodamus $NEURODAMUS_COMMIT

RUN --mount=type=bind,source=ci/scripts/build-neurodamus-models.sh,target=/tmp/build-neurodamus-models.sh \
    --mount=type=cache,target=/var/cache/sccache \
    source /tmp/build-neurodamus-models.sh \
    && source /workspace/user_venv/bin/activate \
    && build-neocortex-models $NEURODAMUS_MODELS_COMMIT

RUN --mount=type=bind,source=ci/scripts/make-env.sh,target=/tmp/make-env.sh \
    source /workspace/user_venv/bin/activate \
    && source /tmp/make-env.sh \
    && make-env
