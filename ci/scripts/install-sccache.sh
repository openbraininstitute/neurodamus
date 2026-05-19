#!/bin/bash

install-sccache() {
    local VERSION=v0.15.0
	local ARCH=$(uname -m)

    set -x
    if [[ -n $SCCACHE_DIR ]]; then
 	 curl -fsSL https://github.com/mozilla/sccache/releases/download/${VERSION}/sccache-${VERSION}-${ARCH}-unknown-linux-musl.tar.gz \
 		 | $SUDO tar xz --strip-components=1 -C /usr/bin sccache-${VERSION}-${ARCH}-unknown-linux-musl/sccache
    fi
    sccache --version
}
