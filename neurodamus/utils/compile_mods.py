#!/usr/bin/env python3
import logging
import argparse
import hashlib
import json
import platform
import shutil
import subprocess  # noqa: S404
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.util import find_spec
from pathlib import Path

import libsonata


L = logging.getLogger(__name__)
VERSION = 1


class Simulator(Enum):
    neuron = "neuron"
    coreneuron = "coreneuron"


@dataclass
class Options:
    simulator: Simulator
    incflags: str
    loadflags: str


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
                L.warning("Already seen `%s` (%s), overriding with `%s`",
                          f.name, f.absolute(), files[f.name])
            files[f.name] = f.absolute()

    hashed_files = {}
    seen_hashes = set()
    for p in files.values():
        hash = _md5sum(p)
        if hash in seen_hashes:
            L.warning("Already added a file with the same contents: %s", hash)
        seen_hashes.add(hash)
        hashed_files[p] = hash
    return hashed_files


def _generate_mod_metadata(mod_files: dict[Path, str], options: Options) -> dict:
    """Create metadata about the compiled mod files, used to track cache hit."""

    def factory(pairs):
        return {k: v.value if isinstance(v, Enum) else v for k, v in pairs}

    return {
        "version": VERSION,
        "hashes": [[p.name, hash_] for p, hash_ in mod_files.items()],
        **asdict(options, dict_factory=factory),
    }


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


def _build_mod_files(
    input_dirs: list[Path], output_dir: Path, nrnivmodl_path: str | None, options: Options
) -> dict:
    """Compile the mod files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    mod_files = _get_mod_files(input_dirs)
    if not mod_files:
        msg = "No mod files selected to be compiled"
        raise RuntimeError(msg)

    if _check_cache(mod_files, output_dir, options):
        return _output_files(output_dir, options)

    nrnivmodl = nrnivmodl_path or shutil.which("neurodamus-nrnivmodl") or shutil.which("nrnivmodl")
    if not nrnivmodl:
        msg = "nrnivmodl not found in PATH"
        raise RuntimeError(msg)

    cmd = [nrnivmodl]

    match options.simulator:
        case Simulator.coreneuron:
            cmd.append("-coreneuron")

    if options.incflags:
        cmd.extend(["-incflags", f"{options.incflags}"])

    if options.loadflags:
        cmd.extend(["-loadflags", options.loadflags])

    mod_dir = (output_dir / "mod_files").absolute()
    mod_dir.mkdir(exist_ok=True)
    for path in mod_files:
        shutil.copy(path, mod_dir)

    cmd.append(str(mod_dir))

    res = subprocess.run(cmd, cwd=str(output_dir), stdout=sys.stderr, check=False)  # noqa: S603

    if res.returncode:
        raise RuntimeError("Failed to compile")

    _write_cache(mod_files, output_dir, options)

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
        "--simulator", type=Simulator, choices=list(Simulator), default=Simulator.neuron
    )
    parser.add_argument("--incflags", help="`--incflags` passed to nrnivmodl")
    parser.add_argument("--loadflags", help="`--loadflags` passed to nrnivmodl")
    parser.add_argument("--nrnivmodl", help="Optional path to nrnivmodl")
    parser.add_argument(
        "--output-type", choices=["json", "shell"], default="json", help="Output type"
    )
    return parser


def compile_mods():
    args = build_parser().parse_args()
    input_dirs = [Path(d).absolute() for d in args.input_dir] if args.input_dir else []
    output_dir = Path(args.output_dir)

    if args.circuit_config:
        input_dirs.extend(_extract_mechanisms_dir(Path(args.circuit_config)))

    if args.with_internal_mods:
        input_dirs.extend(_internal_mods_path())

    options = Options(incflags=args.incflags, loadflags=args.loadflags, simulator=args.simulator)
    env = _build_mod_files(input_dirs, output_dir, args.nrnivmodl, options)

    match args.output_type:
        case "json":
            print(json.dumps(env))  # noqa: T201
        case "shell":
            print("\n".join(f"{k}={v}" for k, v in env.items()))  # noqa: T201


if __name__ == "__main__":
    compile_mods()
