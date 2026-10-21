#!/usr/bin/env python3
import argparse
import errno
import hashlib
import json
import logging
import os
import platform
import secrets
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.util import find_spec
from pathlib import Path

import libsonata

L = logging.getLogger(__name__)
VERSION = 2

MOD_FILES_PATH = "mod_files"

# Ephemeral dirs live beside entries (same filesystem) so rename publish/delete works.
_STAGING_PREFIX = ".tmp-"
_TO_DELETE_PREFIX = ".to-delete-"


class Simulator(Enum):
    neuron = "neuron"
    coreneuron = "coreneuron"


@dataclass(kw_only=True)
class Options:
    simulator: Simulator
    incflags: str
    loadflags: str
    # Included to invalidate ABI-incompatible libraries.
    toolchain: str


def _metadata_path(cache_entry: Path):
    """Return the metadata path inside a cache entry."""
    return cache_entry / "modules.json"


def _md5sum(path: Path) -> str:
    """Get the md5sum of the contents of `path`."""
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_mod_files(input_dirs: list[Path]) -> dict[Path, str]:
    """Get all the mod files, with their md5_sum.

    Note: files with the same names; the last one "wins"
    """
    files = {}
    for d in input_dirs:
        for f in d.glob("*.mod"):
            if f.name in files:
                L.warning(
                    "Already seen `%s` (%s), overriding with `%s`",
                    f.name,
                    files[f.name],
                    f.absolute(),
                )
            files[f.name] = f.absolute()

    hashed_files = {}
    seen_hashes = set()
    for p in files.values():
        hashed_files[p] = _md5sum(p)
        if hashed_files[p] in seen_hashes:
            L.warning("Already added a file with the same contents: %s", hashed_files[p])
        seen_hashes.add(hashed_files[p])
    return hashed_files


def _generate_mod_metadata(mod_files: dict[Path, str], options: Options) -> dict:
    """Create metadata about the compiled mod files, used to track cache hit."""

    def factory(pairs):
        return {k: v.value if isinstance(v, Enum) else v for k, v in pairs}

    return {
        "version": VERSION,
        # Canonical order keeps equivalent file sets on one cache key.
        "hashes": sorted([p.name, hash_] for p, hash_ in mod_files.items()),
        **asdict(options, dict_factory=factory),
    }


def _neuron_version() -> str:
    """Return the installed NEURON version.

    Temporarily clears ``NEURON_INIT_MPI`` so import does not require an MPI library;
    compile-mods only needs the version string for the cache key.
    """
    saved = os.environ.pop("NEURON_INIT_MPI", None)
    try:
        import neuron
    except ImportError as error:
        msg = "neuron package is required"
        raise RuntimeError(msg) from error
    finally:
        if saved is not None:
            os.environ["NEURON_INIT_MPI"] = saved
    return neuron.__version__


def _toolchain_id(nrnivmodl: str) -> str:
    """Return the toolchain identity used in the cache key.

    `nrnivmodl` must already exist; call `_resolve_nrnivmodl` first.
    Versioned identities are shared across virtual environments.
    """
    compiler_path = Path(nrnivmodl).resolve()
    return "|".join(
        [
            platform.machine(),
            sys.platform,
            compiler_path.name,
            f"neuron-{_neuron_version()}",
        ]
    )


def _resolve_nrnivmodl(nrnivmodl_path: str | None) -> str:
    """Find the compiler to use, preferring an explicit path, then the neurodamus wrapper."""
    nrnivmodl = nrnivmodl_path or shutil.which("neurodamus-nrnivmodl") or shutil.which("nrnivmodl")
    if not nrnivmodl:
        msg = "nrnivmodl not found in PATH"
        raise RuntimeError(msg)

    path = Path(nrnivmodl)
    if not path.is_file():
        msg = f"nrnivmodl not found: {path}"
        raise RuntimeError(msg)
    if not os.access(path, os.X_OK):
        msg = f"nrnivmodl is not executable: {path}"
        raise RuntimeError(msg)
    return str(path.resolve())


def _cache_entry_key(mod_files: dict[Path, str], options: Options) -> str:
    """Content hash identifying a build: the mod files plus all compilation options."""
    metadata = _generate_mod_metadata(mod_files, options)
    blob = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _cache_entry_path(cache_dir: Path, key: str) -> Path:
    """Return the cache entry for `key`."""
    return cache_dir / key[:2] / key[2:4] / key[4:]


def _staging_dir(cache_entry: Path) -> Path:
    """Directory for an in-progress build of `cache_entry`, on the same filesystem."""
    return cache_entry.parent / (
        f"{_STAGING_PREFIX}{cache_entry.name}-{os.getpid()}-{secrets.token_hex(8)}"
    )


def _to_delete_dir(cache_entry: Path) -> Path:
    """Sibling path used to atomically take `cache_entry` out of the cache namespace."""
    return cache_entry.parent / (
        f"{_TO_DELETE_PREFIX}{cache_entry.name}-{os.getpid()}-{secrets.token_hex(8)}"
    )


def _remove_cache_dir(cache_entry: Path) -> bool:
    """Unpublish `cache_entry` via rename, then delete. Returns whether anything was removed."""
    if not cache_entry.exists():
        return False

    removed = _to_delete_dir(cache_entry)
    try:
        cache_entry.rename(removed)
    except FileNotFoundError:
        return False
    except OSError as error:
        L.warning("Could not unpublish `%s` for removal: %s", cache_entry, error)
        return False

    shutil.rmtree(removed, ignore_errors=True)
    return True


def _publish_cache_entry(staging: Path, cache_entry: Path) -> bool:
    """Atomically publish a complete staging build to `cache_entry`.

    Returns True if this process published `staging`. Returns False if `cache_entry`
    already exists; `staging` is left in place for the caller to retry or discard.
    """
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.rename(cache_entry)
    except OSError as error:
        # Linux: renaming onto an existing directory yields EEXIST / ENOTEMPTY.
        if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} and not cache_entry.exists():
            raise
        return False
    return True


def _touch_cache_entry(cache_entry: Path):
    """Refresh cache-entry mtime for external cache management; failures are logged."""
    try:
        # No Path equivalent that refreshes mtime without creating a missing path.
        os.utime(cache_entry)
    except OSError as error:
        L.warning("Could not update cache entry mtime for `%s`: %s", cache_entry, error)


def _get_dynamic_file(cache_entry: Path, name: str) -> Path:
    """Return the path of a compiled library inside a cache entry."""
    base = (cache_entry / platform.machine()).absolute()
    ext = ".dylib" if sys.platform == "darwin" else ".so"

    return base / f"{name}{ext}"


def _is_cache_entry_valid(mod_files: dict[Path, str], cache_entry: Path, options: Options) -> bool:
    """Return True if `cache_entry` is a complete build for this mod set and options."""
    metadata_path = _metadata_path(cache_entry)
    if not metadata_path.exists():
        return False

    try:
        old = json.loads(metadata_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

    new = _generate_mod_metadata(mod_files, options)

    if old != new:
        return False

    libnrnmech = _get_dynamic_file(cache_entry, "libnrnmech")

    if not libnrnmech.exists():
        return False

    if options.simulator == Simulator.coreneuron:
        coreneuronlib = _get_dynamic_file(cache_entry, "libcorenrnmech")
        if not coreneuronlib.exists():
            return False

    return True


def _write_cache(mod_files: dict[Path, str], cache_entry: Path, options: Options):
    """Write build metadata into `cache_entry`."""
    with _metadata_path(cache_entry).open("w") as fd:
        json.dump(_generate_mod_metadata(mod_files, options), fd)


def _place_mod_files(output_dir: Path, mod_files: Iterable[Path]):
    """Place mod files in the output directory."""
    mod_dir = (output_dir / MOD_FILES_PATH).absolute()
    mod_dir.mkdir(exist_ok=True)
    for f in mod_dir.glob("*.mod"):
        if f.is_file():
            f.unlink()

    for path in mod_files:
        shutil.copy(path, mod_dir)

    return mod_dir


def _compile(
    mod_files: dict[Path, str], output_dir: Path, nrnivmodl: str, options: Options
) -> dict[str, str]:
    """Run the compiler and record the cache metadata on success."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep long builds from looking stale to age-based cache cleanup.
    _touch_cache_entry(output_dir)

    cmd = [nrnivmodl]

    match options.simulator:
        case Simulator.coreneuron:
            cmd.append("-coreneuron")

    if options.incflags:
        cmd.extend(["-incflags", f"{options.incflags}"])

    if options.loadflags:
        cmd.extend(["-loadflags", options.loadflags])

    mod_dir = _place_mod_files(output_dir, mod_files.keys())

    cmd.append(str(mod_dir))

    L.info("Compiling %d mod file(s) into %s", len(mod_files), output_dir)
    res = subprocess.run(cmd, cwd=str(output_dir), stdout=sys.stderr, check=False)  # ruff: ignore[subprocess-without-shell-equals-true]

    if res.returncode:
        raise RuntimeError("Failed to compile")

    # Written last so it also marks a successful build.
    _write_cache(mod_files, output_dir, options)

    return _output_files(output_dir, options)


def _copy_cache_entry(cache_entry: Path, output_dir: Path):
    """Copy a compiled cache entry into the requested output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in cache_entry.iterdir():
        destination = output_dir / source.name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _discard_staging(staging: Path, cache_entry: Path) -> None:
    """Drop a staging build that lost the publish race to a valid cache entry."""
    L.debug("Cache entry already published, discarding staging build: %s", cache_entry)
    shutil.rmtree(staging, ignore_errors=True)


def _try_publish_staging(
    staging: Path,
    cache_entry: Path,
    mod_files: dict[Path, str],
    options: Options,
) -> str | None:
    """Publish or reuse `cache_entry`. Returns an error message, or None on success."""
    if _publish_cache_entry(staging, cache_entry):
        return None
    if _is_cache_entry_valid(mod_files, cache_entry, options):
        _discard_staging(staging, cache_entry)
        return None

    # Occupied by an invalid leftover (remove may have failed earlier); retry once.
    L.info("Removing invalid cache entry before retrying publish: %s", cache_entry)
    if not _remove_cache_dir(cache_entry):
        return (
            f"Could not publish cache entry: invalid leftover at `{cache_entry}` "
            "could not be removed"
        )

    if _publish_cache_entry(staging, cache_entry):
        return None
    if _is_cache_entry_valid(mod_files, cache_entry, options):
        _discard_staging(staging, cache_entry)
        return None

    return (
        f"Could not publish cache entry: `{cache_entry}` is still missing or invalid after compile"
    )


def _compile_into_cache(
    mod_files: dict[Path, str],
    cache_entry: Path,
    nrnivmodl: str,
    options: Options,
) -> None:
    """Build into a staging directory and atomically publish to `cache_entry`."""
    if cache_entry.exists() and not _is_cache_entry_valid(mod_files, cache_entry, options):
        # Incomplete leftovers cannot be replaced by rename; unpublish first.
        L.info("Removing invalid cache entry before rebuild: %s", cache_entry)
        _remove_cache_dir(cache_entry)

    if _is_cache_entry_valid(mod_files, cache_entry, options):
        return

    staging = _staging_dir(cache_entry)
    try:
        _compile(mod_files, staging, nrnivmodl, options)
        error = _try_publish_staging(staging, cache_entry, mod_files, options)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if error:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(error)


def _materialize_cache_entry(
    mod_files: dict[Path, str],
    cache_entry: Path,
    output_dir: Path,
    nrnivmodl: str,
    options: Options,
) -> dict[str, str]:
    """Copy `cache_entry` into `output_dir`, rebuilding once if it disappears."""
    _touch_cache_entry(cache_entry)
    try:
        _copy_cache_entry(cache_entry, output_dir)
    except FileNotFoundError:
        L.warning("Cache entry disappeared while copying; rebuilding: %s", cache_entry)
        _compile_into_cache(mod_files, cache_entry, nrnivmodl, options)
        if not _is_cache_entry_valid(mod_files, cache_entry, options):
            msg = (
                f"Cache entry missing or invalid after compile: `{cache_entry}`. "
                "Another process may have left an incomplete entry that could not be replaced."
            )
            raise RuntimeError(msg) from None
        _touch_cache_entry(cache_entry)
        _copy_cache_entry(cache_entry, output_dir)
    return _output_files(output_dir, options)


def _build_mod_files(
    input_dirs: list[Path],
    output_dir: Path,
    nrnivmodl: str,
    options: Options,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Compile MOD files into `output_dir`, optionally via a shared `--cache-dir`."""
    mod_files = _get_mod_files(input_dirs)
    if not mod_files:
        msg = "No mod files selected to be compiled"
        raise RuntimeError(msg)

    if cache_dir is None:
        return _compile(mod_files, output_dir, nrnivmodl, options)

    cache_entry = _cache_entry_path(cache_dir, _cache_entry_key(mod_files, options))

    if _is_cache_entry_valid(mod_files, cache_entry, options):
        L.info("Cache hit, materializing: %s", cache_entry)
        return _materialize_cache_entry(mod_files, cache_entry, output_dir, nrnivmodl, options)

    # Miss: compile under staging, rename into place. Concurrent misses may all compile;
    # only one rename wins and the others reuse the published cache entry.
    _compile_into_cache(mod_files, cache_entry, nrnivmodl, options)

    if not _is_cache_entry_valid(mod_files, cache_entry, options):
        msg = (
            f"Cache entry missing or invalid after compile: `{cache_entry}`. "
            "Another process may have left an incomplete entry that could not be replaced."
        )
        raise RuntimeError(msg)

    return _materialize_cache_entry(mod_files, cache_entry, output_dir, nrnivmodl, options)


def _output_files(output_dir: Path, options: Options) -> dict[str, str]:
    """Returned the output shared object files."""
    base = (output_dir / platform.machine()).absolute()
    ext = ".dylib" if sys.platform == "darwin" else ".so"

    libnrnmech = base / f"libnrnmech{ext}"
    if not libnrnmech.exists():
        msg = f"{libnrnmech} does not exist, error running nrnivmodl?"
        raise RuntimeError(msg)

    ret = {"NRNMECH_LIB_PATH": str(libnrnmech), "SPECIALS_PATH": str(base)}

    if options.simulator == Simulator.coreneuron:
        coreneuronlib = base / f"libcorenrnmech{ext}"
        if not coreneuronlib.exists():
            msg = f"{coreneuronlib} does not exist, error running nrnivmodl?"
            raise RuntimeError(msg)

        ret["CORENEURONLIB"] = str(coreneuronlib)

    return ret


def _extract_mechanisms_dir(circuit_config_path: Path) -> list[Path]:
    """Get `mechanisms_dir` from SONATA config."""
    cc = libsonata.CircuitConfig.from_file(circuit_config_path)
    paths = set()
    for name in cc.node_populations:
        properties = cc.node_population_properties(name)
        # libsonata < 0.1.37 omits this attribute.
        if d := getattr(properties, "mechanisms_dir", None):
            paths.add(Path(d).absolute())
    return list(paths)


def _internal_mods_path() -> list[Path]:
    """Get path to the internal neurodamus mods."""
    spec = find_spec("neurodamus")
    assert spec
    assert spec.origin
    path = Path(spec.origin).parent / "data/mod/"
    return [path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-internal-mods", action="store_true", help="Include neurodamus internal mods"
    )
    parser.add_argument("--circuit-config", help="Circuit config for `mechanisms_dir` discovery")
    parser.add_argument("--input-dir", nargs="*", help="Input directory")
    parser.add_argument("--output-dir", required=True, help="Output path for compiled mods")
    parser.add_argument(
        "--cache-dir",
        help="Cache root; matching entries are copied into `--output-dir`",
    )
    parser.add_argument(
        "--simulator", type=Simulator, choices=list(Simulator), default=Simulator.neuron
    )
    parser.add_argument("--incflags", help="`--incflags` passed to nrnivmodl")
    parser.add_argument("--loadflags", help="`--loadflags` passed to nrnivmodl")
    parser.add_argument("--nrnivmodl", help="Optional path to nrnivmodl")
    parser.add_argument(
        "--output-type", choices=["json", "shell"], default="json", help="Output type"
    )
    return parser


def _setup_cli_logging():
    """Configure isolated stderr logging for the CLI."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    L.handlers[:] = [handler]
    L.setLevel(logging.INFO)
    L.propagate = False


def compile_mods():
    _setup_cli_logging()

    parser = build_parser()
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).absolute() if args.cache_dir else None
    output_dir = Path(args.output_dir)

    input_dirs = [Path(d).absolute() for d in args.input_dir] if args.input_dir else []

    if args.circuit_config:
        input_dirs.extend(_extract_mechanisms_dir(Path(args.circuit_config)))

    if args.with_internal_mods:
        input_dirs.extend(_internal_mods_path())

    nrnivmodl = _resolve_nrnivmodl(args.nrnivmodl)
    # argparse leaves omitted string flags as None; normalize so cache keys stay stable.
    options = Options(
        incflags=args.incflags or "",
        loadflags=args.loadflags or "",
        simulator=args.simulator,
        toolchain=_toolchain_id(nrnivmodl),
    )

    env = _build_mod_files(
        input_dirs,
        output_dir,
        nrnivmodl,
        options,
        cache_dir=cache_dir,
    )

    match args.output_type:
        case "json":
            print(json.dumps(env))  # ruff: ignore[print]
        case "shell":
            print("\n".join(f"{k}={v}" for k, v in env.items()))  # ruff: ignore[print]


if __name__ == "__main__":
    compile_mods()
