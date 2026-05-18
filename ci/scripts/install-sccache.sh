#!/bin/bash

install-sccache() {
    local VERSION=v0.15.0
	local ARCH=$(uname -m)

    if [[ -n $SCCACHE_DIR ]]; then
 	 curl -fsSL https://github.com/mozilla/sccache/releases/download/${VERSION}/sccache-${VERSION}-${ARCH}-unknown-linux-musl.tar.gz \
 		 | tar xz --strip-components=1 -C /usr/bin sccache-${VERSION}-${ARCH}-unknown-linux-musl/sccache
    fi
    sccache --version
}
