#!/usr/bin/env bash
# Clone and pip-install neurodamus
#
# Environment variables:
#   PRE       called on function entry
#   BUILD_DIR Directory for cloning source (e.g. /tmp)
#   PIP       pip command to use (e.g. "uv pip")

build-neurodamus() {
    PRE || true

    : "${PIP:?PIP is not set}"

    local COMMIT=${1:-HEAD}

    local NEURODAMUS=$BUILD_DIR/neurodamus

    if [[ ! -e $NEURODAMUS ]]; then
       git clone --filter=blob:none \
        https://github.com/openbraininstitute/neurodamus/ $NEURODAMUS
    fi

    ( cd $NEURODAMUS && \
        git fetch --depth 1 origin $COMMIT &&
        git checkout FETCH_HEAD
    )

    $PIP install $NEURODAMUS'[full]'
}
