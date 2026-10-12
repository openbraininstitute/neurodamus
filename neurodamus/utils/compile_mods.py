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
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.util import find_spec
from pathlib import Path

import libsonata

L = logging.getLogger(__name__)
VERSION = 2

MOD_FILES_PATH = "mod_files"

LOCK_FILE = ".lock"

# Five minutes covers the measured CI MOD builds without masking a stuck compiler.
DEFAULT_LOCK_TIMEOUT_SECONDS = 5 * 60

_LOCK_POLL_MIN_SECONDS = 0.05
_LOCK_POLL_MAX_SECONDS = 5.0
_LOCK_HEARTBEAT_SECONDS = 60.0

NEURON_DISTRIBUTIONS = ("NEURON", "neuron", "neuron-nightly")

# Default age for `--clear-cache`.
DEFAULT_MAX_AGE_DAYS = 30.0


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
    """Return the installed NEURON version, or None if it cannot be determined.

    Uses distribution metadata rather than `import neuron`, which is slow and has
    side effects. NEURON installed outside of python packaging (ie: spack, modules)
    has no metadata; `_toolchain_id` falls back to the compiler path in that case.
    """
    for dist in NEURON_DISTRIBUTIONS:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:  # ruff: ignore[try-except-in-loop] - independent fallback
            continue
    return None


def _toolchain_id(nrnivmodl: str) -> str:
    """Identity of the toolchain used to build the mod files.

    Included in the cache key so that changing NEURON invalidates previously compiled
    libraries, which would otherwise be loaded despite being ABI-incompatible.
    """
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


def _cache_key(mod_files: dict[Path, str], options: Options) -> str:
    """Content hash identifying a build: the mod files plus all compilation options."""
    metadata = _generate_mod_metadata(mod_files, options)
    blob = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _cache_slot(cache_dir: Path, key: str) -> Path:
    """Directory holding the build for `key`.

    Fanned out over two levels so no single directory grows large, which matters on the
    shared filesystems a cluster-wide cache lives on.
    """
    return cache_dir / key[:2] / key[2:4] / key[4:]


def _acquire_lock(fd, lock_path: Path, timeout: float) -> bool:
    """Take an exclusive lock on `fd`, waiting at most `timeout` seconds (0 waits forever).

    Returns whether the lock is actually held: on filesystems that do not implement
    `flock` (ie: Lustre mounted without `-o flock`) we report and carry on unlocked,
    rather than making the tool unusable there.

    Raises:
        RuntimeError: the lock could not be taken within `timeout`.
    """
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    started = time.monotonic()
    next_report = started
    delay = _LOCK_POLL_MIN_SECONDS

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        except OSError as e:
            L.warning(
                "Locking is unsupported on `%s` (%s); building without it. "
                "Concurrent builds of this cache slot may corrupt it.",
                lock_path,
                e.strerror,
            )
            return False
        else:
            return True

        now = time.monotonic()
        if now >= next_report:
            L.info(
                "Waiting for another build to release `%s` (%.0fs elapsed)",
                lock_path,
                now - started,
            )
            next_report = now + _LOCK_HEARTBEAT_SECONDS

        if deadline is not None and now >= deadline:
            msg = (
                f"Timed out after {now - started:.0f}s waiting for `{lock_path}`. "
                "Another process is likely stuck compiling: check for a running nrnivmodl, "
                "and delete the lock file if none is. Use --lock-timeout to wait longer, "
                "or 0 to wait indefinitely."
            )
            raise RuntimeError(msg)

        if deadline is not None:
            delay = min(delay, max(deadline - now, 0.0))
        time.sleep(delay)
        delay = min(max(delay * 2, _LOCK_POLL_MIN_SECONDS), _LOCK_POLL_MAX_SECONDS)


@contextmanager
def _slot_lock(slot: Path, timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS):
    """Hold an exclusive lock on a cache slot while building it.

    `flock` is released by the kernel if the holder dies, so a killed build leaves no
    stale lock; it only leaves a slot without `modules.json`, which reads as a miss.
    A holder that hangs rather than dies does block others, hence `timeout`.
    """
    slot.mkdir(parents=True, exist_ok=True)
    lock_path = slot / LOCK_FILE
    with open(lock_path, "ab") as fd:
        locked = _acquire_lock(fd, lock_path, timeout)
        try:
            yield
        finally:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)


def _slot_in_use(slot: Path) -> bool:
    """Whether another process is currently building `slot`."""
    lock_path = slot / LOCK_FILE
    if not lock_path.exists():
        return False
    with open(lock_path, "ab") as fd:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError as e:
            # Unknown lock state is treated as in use.
            L.debug("Cannot test `%s` (%s); assuming in use", lock_path, e.strerror)
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
    return False


def _touch_slot(slot: Path):
    """Refresh the slot mtime, so that ``--clear-cache`` can evict by least recent use."""
    try:
        os.utime(slot)
    except OSError as e:
        L.debug("Could not update mtime of %s: %s", slot, e)


def _clear_cache(cache_dir: Path, max_age_days: float) -> list[Path]:
    """Remove cache slots unused for longer than `max_age_days`. Slots being built are kept."""
    cutoff = time.time() - max_age_days * 86400
    removed = []
    for slot in sorted(cache_dir.glob("*/*/*")):
        if not slot.is_dir():
            continue
        if slot.stat().st_mtime >= cutoff:
            continue
        if _slot_in_use(slot):
            L.info("Cache slot in use, keeping: %s", slot)
            continue
        L.info("Removing cache slot: %s", slot)
        shutil.rmtree(slot, ignore_errors=True)
        removed.append(slot)
    return removed


def _get_dynamic_file(output_dir: Path, name: str) -> Path:
    """Returns the name of the file with the correct machine directory for NEURON."""
    base = (output_dir / platform.machine()).absolute()
    ext = ".dylib" if sys.platform == "darwin" else ".so"

    return base / f"{name}{ext}"


def _check_cache(mod_files: dict[Path, str], output_dir: Path, options: Options) -> bool:
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
    # Keep rebuilding slots from being cleared.
    _touch_slot(output_dir)

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


def _build_mod_files(
    input_dirs: list[Path],
    output_dir: Path | None,
    nrnivmodl: str,
    options: Options,
    cache_dir: Path | None = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> dict[str, str]:
    """Compile the mod files, reusing a previous build when nothing relevant changed.

    With `cache_dir`, the build location is derived from a hash of the mod files and the
    compilation options, so unrelated builds coexist and identical ones are shared.
    """
    mod_files = _get_mod_files(input_dirs)
    if not mod_files:
        msg = "No mod files selected to be compiled"
        raise RuntimeError(msg)

    if cache_dir is not None:
        output_dir = _cache_slot(cache_dir, _cache_key(mod_files, options))
        L.debug("Cache slot: %s", output_dir)

    assert output_dir is not None, "either output_dir or cache_dir is required"

    # Fast path avoids locking on cache hits.
    if _check_cache(mod_files, output_dir, options):
        L.info("Cache hit, skipping compilation: %s", output_dir)
        _touch_slot(output_dir)
        return _output_files(output_dir, options)

    lock = _slot_lock(output_dir, lock_timeout) if cache_dir is not None else nullcontext()
    with lock:
        # Recheck after waiting for another builder.
        if _check_cache(mod_files, output_dir, options):
            L.info("Cache hit after lock, skipping compilation: %s", output_dir)
            _touch_slot(output_dir)
            return _output_files(output_dir, options)

        return _compile(mod_files, output_dir, nrnivmodl, options)


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
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", help="Output path")
    destination.add_argument(
        "--cache-dir",
        help="Cache root; the output path is derived from a hash of the mod files and options",
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
        "--print-cache-path",
        action="store_true",
        help="Print the resolved output path and exit, without compiling",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Remove stale slots from `--cache-dir` and exit, without compiling",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="With `--cache-dir`, how long to wait for a concurrent build of the same slot "
        "before failing. 0 waits indefinitely",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=DEFAULT_MAX_AGE_DAYS,
        help="With `--clear-cache`, the age past which an unused cache slot is removed",
    )
    return parser


def _setup_cli_logging():
    """Send this module's logs to stderr, regardless of any ambient logging setup.

    Importing `neurodamus` installs a root handler and leaves the root level at WARNING,
    so `basicConfig` is a no-op here. Logging on a dedicated, non-propagating handler also
    guarantees nothing lands on stdout, which carries the machine-readable output.
    """
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
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.clear_cache:
        if cache_dir is None:
            parser.error("--clear-cache requires --cache-dir")
        removed = _clear_cache(cache_dir, args.max_age_days)
        L.info("Cleared %d cache slot(s) from %s", len(removed), cache_dir)
        return

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

    if args.print_cache_path:
        mod_files = _get_mod_files(input_dirs)
        if not mod_files:
            msg = "No mod files selected to be compiled"
            raise RuntimeError(msg)
        resolved = (
            _cache_slot(cache_dir, _cache_key(mod_files, options))
            if cache_dir is not None
            else output_dir
        )
        print(resolved)  # ruff: ignore[print]
        return

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
