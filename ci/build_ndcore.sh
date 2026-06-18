#!/usr/bin/env bash

set -x
set -euo pipefail

CORENEURON=false
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --coreneuron) CORENEURON=true; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

set -- "${ARGS[@]}"

BUILD_DIR="$1"
MODS_DIR="${2:-}"

CMD=(
    neurodamus-compile-mods
    --input-dir $MODS_DIR
    --output-dir $BUILD_DIR
    --internal-mods
    --incflags="-DDISABLE_REPORTINGLIB"
    --output-type shell
    )

if [ "$CORENEURON" = true ]; then
    CMD+=(--simulator=coreneuron)
fi

exports=$(${CMD[@]})

while IFS= read -r line; do
    export "$line"
done <<< "$exports"

HOC_LIBRARY_PATH="$(dirname "$0")"/../neurodamus/data/hoc

echo "export HOC_LIBRARY_PATH=$HOC_LIBRARY_PATH" > $BUILD_DIR/.envfile
echo "export NRNMECH_LIB_PATH=$NRNMECH_LIB_PATH" >> $BUILD_DIR/.envfile
echo "export PATH=$SPECIALS_PATH:\$PATH" >> $BUILD_DIR/.envfile

if [ "$CORENEURON" = true ]; then
    echo "export CORENEURONLIB=$CORENEURONLIB" >> $BUILD_DIR/.envfile
fi
