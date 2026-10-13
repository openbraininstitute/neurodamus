#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.util import find_spec
from pathlib import Path

import libsonata

L = logging.getLogger(__name__)
VERSION = 2

MOD_FILES_PATH = "mod_files"

LOCK_SUFFIX = ".lock"

# Two minutes leaves CI headroom without masking a stuck compiler.
DEFAULT_LOCK_TIMEOUT_SECONDS = 2 * 60

_LOCK_POLL_MIN_SECONDS = 0.05
_LOCK_POLL_MAX_SECONDS = 1.0
_LOCK_HEARTBEAT_SECONDS = 60.0

NEURON_DISTRIBUTION = "neuron"

# Owner-only lock files; the same UID runs all tasks.
_LOCK_FILE_MODE = 0o600

# Default age for `--clear-cache`.
DEFAULT_MAX_AGE_DAYS = 30.0


class CacheLockTimeoutError(RuntimeError):
    """Raised when another build holds a cache lock for too long."""


class Simulator(Enum):
    neuron = "neuron"
    coreneuron = "coreneuron"


@dataclass
class Options:
    simulator: Simulator
    incflags: str
    loadflags: str
    # Included to invalidate ABI-incompatible libraries.
    toolchain: str = ""


def _metadata_path(output_dir: Path):
    """Return cache metadata path."""
    return output_dir / "modules.json"


def _md5sum(path: Path) -> str:
    """Get the md5sum of the contents of `path`."""
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
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
                    f.absolute(),
                    files[f.name],
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


def _neuron_version() -> str | None:
    """Return the installed NEURON version, if available."""
    try:
        return importlib.metadata.version(NEURON_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def _toolchain_id(nrnivmodl: str) -> str:
    """Return the toolchain identity used in the cache key."""
    compiler_path = Path(nrnivmodl).resolve()
    parts = [platform.machine(), sys.platform, compiler_path.name]

    if version := _neuron_version():
        # Versioned identities are shared across virtual environments.
        parts.append(f"neuron-{version}")
    else:
        try:
            st = compiler_path.stat()
        except OSError:
            parts.append(f"path-{compiler_path}")
            L.warning("Cannot stat `%s`; cache key is based on its path alone", compiler_path)
        else:
            parts.append(f"path-{compiler_path}:size-{st.st_size}:mtime-{int(st.st_mtime)}")

    return "|".join(parts)


def _resolve_nrnivmodl(nrnivmodl_path: str | None) -> str:
    """Find the compiler to use, preferring an explicit path, then the neurodamus wrapper."""
    nrnivmodl = nrnivmodl_path or shutil.which("neurodamus-nrnivmodl") or shutil.which("nrnivmodl")
    if not nrnivmodl:
        msg = "nrnivmodl not found in PATH"
        raise RuntimeError(msg)
    return nrnivmodl


def _cache_entry_key(mod_files: dict[Path, str], options: Options) -> str:
    """Content hash identifying a build: the mod files plus all compilation options."""
    metadata = _generate_mod_metadata(mod_files, options)
    blob = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _cache_entry_path(cache_dir: Path, key: str) -> Path:
    """Return the cache entry for `key`."""
    return cache_dir / key[:2] / key[2:4] / key[4:]


def _cache_lock_path(entry: Path) -> Path:
    """Lock beside the entry, so clearing the entry cannot unlink a held lock."""
    return entry.parent / f"{entry.name}{LOCK_SUFFIX}"


def _open_lock_file(lock_path: Path) -> int:
    """Open the lock file, keeping it usable by other users of a shared cache."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, _LOCK_FILE_MODE)
    os.fchmod(fd, _LOCK_FILE_MODE)
    return fd


def _lock_waits(lock_path: Path, timeout: float) -> Iterator[None]:
    """Back off between lock attempts, reporting progress and enforcing `timeout`."""
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    started = time.monotonic()
    next_report = started
    delay = _LOCK_POLL_MIN_SECONDS

    while True:
        now = time.monotonic()
        if now >= next_report:
            L.info(
                "Waiting for another build to release `%s` (%.0fs elapsed)",
                lock_path,
                now - started,
            )
            next_report = now + _LOCK_HEARTBEAT_SECONDS

        if deadline is not None:
            if now >= deadline:
                msg = (
                    f"Timed out after {now - started:.0f}s waiting for `{lock_path}`. "
                    "Another process is likely stuck compiling: check for a running "
                    "nrnivmodl, and delete the lock file if none is. Use --lock-timeout "
                    "to wait longer, or 0 to wait indefinitely."
                )
                raise CacheLockTimeoutError(msg)
            delay = min(delay, max(deadline - now, 0.0))

        time.sleep(delay)
        delay = min(max(delay * 2, _LOCK_POLL_MIN_SECONDS), _LOCK_POLL_MAX_SECONDS)
        yield


@contextmanager
def _cache_lock(
    entry: Path,
    timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    *,
    exclusive: bool = True,
):
    """Lock a cache entry, yielding whether the lock is held.

    Shared locks let concurrent readers reuse an entry; only builds need exclusivity.
    `flock` works on EFS and on Lustre mounted with `flock`; where it is unavailable the
    build continues unlocked rather than failing.
    """
    lock_path = _cache_lock_path(entry)
    operation = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
    waits = _lock_waits(lock_path, timeout)
    locked = False
    fd = _open_lock_file(lock_path)

    try:
        while not locked:
            try:
                fcntl.flock(fd, operation)
            except BlockingIOError:  # ruff: ignore[try-except-in-loop] - polling a lock
                next(waits)
            except OSError as e:
                L.warning(
                    "Locking is unsupported on `%s` (%s); building without it. "
                    "Concurrent builds of this cache entry may corrupt it.",
                    lock_path,
                    e.strerror,
                )
                break
            else:
                locked = True

        yield locked
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _cache_entry_in_use(entry: Path) -> bool:
    """Whether another process is currently using `entry`."""
    lock_path = _cache_lock_path(entry)
    if not lock_path.exists():
        return False
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError as e:
        L.debug("Cannot open `%s` (%s); assuming in use", lock_path, e.strerror)
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError as e:
        # Unknown lock state is treated as in use.
        L.debug("Cannot test `%s` (%s); assuming in use", lock_path, e.strerror)
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _touch_cache_entry(entry: Path):
    """Refresh entry mtime; cleanup bookkeeping is advisory, so failures are logged."""
    try:
        os.utime(entry)
    except OSError as error:
        L.warning(
            "Could not update cache entry mtime for `%s`; cleanup age may be inaccurate: %s",
            entry,
            error,
        )


def _clear_cache(cache_dir: Path, max_age_days: float) -> list[Path]:
    """Remove cache entries older than `max_age_days`."""
    cutoff = time.time() - max_age_days * 86400
    removed = []
    for entry in sorted(cache_dir.glob("*/*/*")):
        if not entry.is_dir():
            continue
        if entry.stat().st_mtime >= cutoff:
            continue
        if _cache_entry_in_use(entry):
            L.info("Cache entry in use, keeping: %s", entry)
            continue
        L.info("Removing cache entry: %s", entry)
        shutil.rmtree(entry, ignore_errors=True)
        removed.append(entry)
    return removed


def _get_dynamic_file(output_dir: Path, name: str) -> Path:
    """Returns the name of the file with the correct machine directory for NEURON."""
    base = (output_dir / platform.machine()).absolute()
    ext = ".dylib" if sys.platform == "darwin" else ".so"

    return base / f"{name}{ext}"


def _is_cache_entry_valid(mod_files: dict[Path, str], output_dir: Path, options: Options) -> bool:
    """See if if we have a cache hit."""
    metadata = _metadata_path(output_dir)
    if not metadata.exists():
        return False

    with metadata.open() as fd:
        old = json.load(fd)

    new = _generate_mod_metadata(mod_files, options)
    if old != new:
        return False

    libnrnmech = _get_dynamic_file(output_dir, "libnrnmech")

    if not libnrnmech.exists():
        return False

    if options.simulator == Simulator.coreneuron:
        coreneuronlib = _get_dynamic_file(output_dir, "libcorenrnmech")
        if not coreneuronlib.exists():
            return False

    return True


def _write_cache(mod_files: dict[Path, str], output_dir: Path, options: Options):
    """Write the cache metadata."""
    with _metadata_path(output_dir).open("w") as fd:
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
    # Keep rebuilding entrys from being cleared.
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


def _build_mod_files(
    input_dirs: list[Path],
    output_dir: Path,
    nrnivmodl: str,
    options: Options,
    cache_dir: Path | None = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> dict[str, str]:
    """Compile MOD files and materialize a matching cache entry."""
    mod_files = _get_mod_files(input_dirs)
    if not mod_files:
        msg = "No mod files selected to be compiled"
        raise RuntimeError(msg)

    if cache_dir is None:
        if _is_cache_entry_valid(mod_files, output_dir, options):
            L.info("Cache hit, skipping compilation: %s", output_dir)
            return _output_files(output_dir, options)
        return _compile(mod_files, output_dir, nrnivmodl, options)

    cache_entry = _cache_entry_path(cache_dir, _cache_entry_key(mod_files, options))

    # Readers only need a shared lock, so concurrent hits are not serialized.
    with _cache_lock(cache_entry, lock_timeout, exclusive=False):
        if _is_cache_entry_valid(mod_files, cache_entry, options):
            L.info("Cache hit, materializing: %s", cache_entry)
            _touch_cache_entry(cache_entry)
            _copy_cache_entry(cache_entry, output_dir)
            return _output_files(output_dir, options)

    with _cache_lock(cache_entry, lock_timeout):
        # Another builder may have finished while the shared lock was released.
        if _is_cache_entry_valid(mod_files, cache_entry, options):
            _touch_cache_entry(cache_entry)
        else:
            _compile(mod_files, cache_entry, nrnivmodl, options)
        _copy_cache_entry(cache_entry, output_dir)

    return _output_files(output_dir, options)


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
        if d := properties.mechanisms_dir:
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
    parser.add_argument("--output-dir", required=True, help="Output path")
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
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear stale entries from `--cache-dir` before compiling",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="With `--cache-dir`, how long to wait for a concurrent build of the same entry "
        "before failing. 0 waits indefinitely",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=DEFAULT_MAX_AGE_DAYS,
        help="With `--clear-cache`, the age past which an unused cache entry is removed",
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

    if args.clear_cache and cache_dir is not None:
        removed = _clear_cache(cache_dir, args.max_age_days)
        L.info("Cleared %d cache entry(s) from %s", len(removed), cache_dir)

    input_dirs = [Path(d).absolute() for d in args.input_dir] if args.input_dir else []

    if args.circuit_config:
        input_dirs.extend(_extract_mechanisms_dir(Path(args.circuit_config)))

    if args.with_internal_mods:
        input_dirs.extend(_internal_mods_path())

    nrnivmodl = _resolve_nrnivmodl(args.nrnivmodl)
    options = Options(
        incflags=args.incflags,
        loadflags=args.loadflags,
        simulator=args.simulator,
        toolchain=_toolchain_id(nrnivmodl),
    )

    env = _build_mod_files(
        input_dirs,
        output_dir,
        nrnivmodl,
        options,
        cache_dir=cache_dir,
        lock_timeout=args.lock_timeout,
    )

    match args.output_type:
        case "json":
            print(json.dumps(env))  # ruff: ignore[print]
        case "shell":
            print("\n".join(f"{k}={v}" for k, v in env.items()))  # ruff: ignore[print]


if __name__ == "__main__":
    compile_mods()
