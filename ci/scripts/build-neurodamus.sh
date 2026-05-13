#!/bin/bash
# export BUILD_DIR=/tmp

build-neurodamus() {
    pre || true

    : "${PIP:?PIP is not set}"

    local COMMIT=${1:-HEAD}

    local NEURODAMUS=$BUILD_DIR/neurodamus

    git clone --filter=blob:none \
        https://github.com/openbraininstitute/neurodamus/ $NEURODAMUS

    ( cd $NEURODAMUS && \
        git fetch --depth 1 origin $COMMIT &&
        git checkout FETCH_HEAD
    )

    $PIP install $NEURODAMUS
}
