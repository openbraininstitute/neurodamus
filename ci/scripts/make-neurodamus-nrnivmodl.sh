#!/usr/bin/env bash
#
# Wrapper of nrnivmodl that adds libsonatareport
#
# Environment variables:
#   INSTALL_DIR   - Prefix where everything is installed (e.g. /opt/obi)
#   VIRTUAL_ENV   - Path to the active Python virtual environment

make-neurodamus-nrnivmodl() {
    local LIB_EXT=$(if [[ $(uname) == Darwin ]]; then echo dylib; else echo so; fi)

    cat > $INSTALL_DIR/neurodamus-nrnivmodl << _EOF
#!/bin/bash
source $VIRTUAL_ENV/bin/activate

EXTRA_INCFLAGS='-isystem $INSTALL_DIR/include'
EXTRA_LOADFLAGS='-Wl,-rpath,$INSTALL_DIR/lib $INSTALL_DIR/lib/libsonatareport.$LIB_EXT'
_EOF

    cat >> $INSTALL_DIR/neurodamus-nrnivmodl << '_EOF'
CORENEURON=""
INCFLAGS=""
LOADFLAGS=""
NMODL=""
NMODLFLAGS=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -coreneuron) CORENEURON=1; shift ;;
    -incflags)   INCFLAGS="$2"; shift 2 ;;
    -loadflags)  LOADFLAGS="$2"; shift 2 ;;
    -nmodl)      NMODL="$2"; shift 2 ;;
    -nmodlflags) NMODLFLAGS="$2"; shift 2 ;;
    -h|--help)   exec nrnivmodl --help ;;
    *)           POSITIONAL+=("$1"); shift ;;
  esac
done

INCFLAGS="$EXTRA_INCFLAGS${INCFLAGS:+ $INCFLAGS}"
LOADFLAGS="$EXTRA_LOADFLAGS${LOADFLAGS:+ $LOADFLAGS}"

ARGS=()
[[ -n $CORENEURON ]] && ARGS+=("-coreneuron")
ARGS+=("-incflags" "$INCFLAGS")
ARGS+=("-loadflags" "$LOADFLAGS")
[[ -n $NMODL ]] && ARGS+=("-nmodl" "$NMODL")
[[ -n $NMODLFLAGS ]] && ARGS+=("-nmodlflags" "$NMODLFLAGS")


exec nrnivmodl "${ARGS[@]}" "${POSITIONAL[@]}"
_EOF

    chmod +x $INSTALL_DIR/neurodamus-nrnivmodl
}
