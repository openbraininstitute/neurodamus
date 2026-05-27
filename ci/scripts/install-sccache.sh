#!/bin/bash
# Download and install the sccache compiler cache binary.
#
# Environment variables:
#   SCCACHE_DIR  - If set, triggers the installation; otherwise skipped
#   SUDO         - Set to "sudo" if elevated privileges are needed for install

install-sccache() {
    local VERSION=v0.15.0

    if [[ -n $SCCACHE_DIR ]]; then
        if [ "$(uname)" = "Darwin" ]; then
            curl -fsSL https://github.com/mozilla/sccache/releases/download/${VERSION}/sccache-${VERSION}-aarch64-apple-darwin.tar.gz \
                | $SUDO tar xz --strip-components=1 -C /usr/local/bin sccache-${VERSION}-aarch64-apple-darwin/sccache
        elif [ "$(uname)" = "Linux" ]; then
            local ARCH=$(uname -m)
            curl -fsSL https://github.com/mozilla/sccache/releases/download/${VERSION}/sccache-${VERSION}-${ARCH}-unknown-linux-musl.tar.gz \
                | $SUDO tar xz --strip-components=1 -C /usr/bin sccache-${VERSION}-${ARCH}-unknown-linux-musl/sccache
        fi
        sccache --version
    fi
}
