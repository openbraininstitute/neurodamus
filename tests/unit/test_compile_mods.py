import argparse
import errno
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

import neurodamus.utils.compile_mods as test_module

Options = test_module.Options
Simulator = test_module.Simulator

REPO_ROOT = Path(__file__).parent.parent.parent


def write(path, contents):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(contents)
    return path


def test__metadata_path(tmp_path):
    assert test_module._metadata_path(tmp_path) == tmp_path / "modules.json"


def test__md5sum(tmp_path):
    path = write(tmp_path / "a.txt", b"hello")
    assert test_module._md5sum(path) == "5d41402abc4b2a76b9719d911017c592"


def test__place_mod_files(tmp_path):
    output_dir = tmp_path
    mod_dir = output_dir / test_module.MOD_FILES_PATH

    mod_dir.mkdir(parents=True)
    (mod_dir / "old.mod").write_text("old")

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.mod").write_text("aaa")
    (src_dir / "b.mod").write_text("bbb")

    result = test_module._place_mod_files(output_dir, [src_dir / "a.mod", src_dir / "b.mod"])

    assert result == mod_dir.absolute()
    assert not (mod_dir / "old.mod").exists()
    assert (mod_dir / "a.mod").read_text() == "aaa"
    assert (mod_dir / "b.mod").read_text() == "bbb"


def test__get_mod_files(tmp_path, caplog):
    d1 = tmp_path / "a"
    d1.mkdir()
    d2 = tmp_path / "b"
    d2.mkdir()

    write(d1 / "x.mod", b"content_x")
    write(d2 / "y.mod", b"content_y")
    write(d2 / "x.mod", b"content_x_override")  # same name in d2 overrides d1
    write(d2 / "z.mod", b"content_y")  # same content as `y.mod`

    with caplog.at_level(logging.WARNING):
        res = test_module._get_mod_files([d1, d2])

    assert "Already seen `x.mod`" in caplog.text
    assert (
        "Already added a file with the same contents: 531ba6ac24a2ebde442987529f07e697"
        in caplog.text
    )
    names = {p.name for p in res}
    assert "x.mod" in names
    assert "y.mod" in names
    assert next(p for p in res if p.name == "x.mod").parent == d2.absolute()


def test__generate_mod_metadata(tmp_path):
    path = write(tmp_path / "a.mod", b"mod content")

    mod_files = {path: "abc123"}
    options = Options(simulator=Simulator.neuron, incflags="-DFOO", loadflags="")

    meta = test_module._generate_mod_metadata(mod_files, options)
    assert meta["version"] == test_module.VERSION
    assert meta["hashes"] == [["a.mod", "abc123"]]
    assert meta["simulator"] == "neuron"
    assert meta["incflags"] == "-DFOO"
    assert not meta["toolchain"]


def test__check_cache_missing(tmp_path):
    options = Options(simulator=Simulator.neuron, incflags="", loadflags="")
    assert not test_module._check_cache({}, tmp_path, options)


def test__write_cache_and_check_cache(tmp_path):
    path = write(tmp_path / "a.mod", b"content")
    mod_files = {path: "hash1"}
    options0 = Options(simulator=Simulator.neuron, incflags="-DX", loadflags="")

    test_module._write_cache(mod_files, tmp_path, options0)
    assert test_module._metadata_path(tmp_path).exists()

    assert not test_module._check_cache(mod_files, tmp_path, options0)

    write(test_module._get_dynamic_file(tmp_path, "libnrnmech"), b"")
    assert test_module._check_cache(mod_files, tmp_path, options0)

    options1 = Options(simulator=Simulator.coreneuron, incflags="-DX", loadflags="")
    test_module._write_cache(mod_files, tmp_path, options1)
    assert not test_module._check_cache(mod_files, tmp_path, options1)

    options2 = Options(simulator=Simulator.neuron, incflags="-DX", loadflags="-different")
    assert not test_module._check_cache(mod_files, tmp_path, options2)


def test__output_files(tmp_path):
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    arch = platform.machine()
    base = tmp_path / arch
    base.mkdir()
    write(base / f"libnrnmech{ext}", b"")

    options = Options(simulator=Simulator.neuron, incflags="", loadflags="")
    res = test_module._output_files(tmp_path, options)
    assert res["NRNMECH_LIB_PATH"] == str(base / f"libnrnmech{ext}")
    assert res["SPECIALS_PATH"] == str(base)
    assert "CORENEURONLIB" not in res

    write(base / f"libcorenrnmech{ext}", b"")
    options = Options(simulator=Simulator.coreneuron, incflags="", loadflags="")
    res = test_module._output_files(tmp_path, options)
    assert res["NRNMECH_LIB_PATH"] == str(base / f"libnrnmech{ext}")
    assert res["SPECIALS_PATH"] == str(base)
    assert res["CORENEURONLIB"] == str(base / f"libcorenrnmech{ext}")


def test__output_files_coreneuron(tmp_path):
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    arch = platform.machine()
    base = tmp_path / arch
    base.mkdir()
    write(base / f"libnrnmech{ext}", b"")
    write(base / f"libcorenrnmech{ext}", b"")

    options = Options(simulator=Simulator.coreneuron, incflags="", loadflags="")
    res = test_module._output_files(tmp_path, options)
    assert "CORENEURONLIB" in res


def test__internal_mods_path():
    paths = test_module._internal_mods_path()
    assert len(paths) == 1
    assert paths[0].name == "mod"
    assert paths[0].exists()


def test__build_mod_filess(tmp_path):
    options = Options(simulator=Simulator.neuron, incflags="", loadflags="")
    input_dirs = []
    with patch("neurodamus.utils.compile_mods.subprocess.run") as mock_run:
        with pytest.raises(RuntimeError, match="No mod files selected to be compiled"):
            test_module._build_mod_files(input_dirs, tmp_path, "echo", options)

        input_dir = tmp_path / "inputs"
        write(input_dir / "a.mod", b"mod file")
        write(test_module._get_dynamic_file(tmp_path, "libnrnmech"), b"")
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        test_module._build_mod_files([input_dir], tmp_path, "echo", options)

        test_module._build_mod_files([input_dir], tmp_path, "echo", options)


def test__build_parser():
    parser = test_module.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test__extract_mechanisms_dir(tmp_path):
    circuit_config = Path(__file__).parent.parent / "simulations/ringtest/circuit_config.json"
    assert test_module._extract_mechanisms_dir(circuit_config) == []

    with circuit_config.open() as fd:
        js = json.load(fd)

    circuit_config = tmp_path / "circuit_config.json"
    js["components"] = {"mechanisms_dir": "path/to/mechs"}
    with circuit_config.open("w") as fd:
        json.dump(js, fd)
    res = test_module._extract_mechanisms_dir(circuit_config)
    assert len(res) == 1
    assert str(res[0]).endswith("path/to/mechs")


def options(**kwargs):
    kwargs.setdefault("simulator", Simulator.neuron)
    kwargs.setdefault("incflags", "")
    kwargs.setdefault("loadflags", "")
    kwargs.setdefault("toolchain", "toolchain-a")
    return Options(**kwargs)


def fake_nrnivmodl(cmd, cwd=None, **_kwargs):
    """Stand in for nrnivmodl: drop a libnrnmech where `_output_files` looks for it."""
    write(test_module._get_dynamic_file(Path(cwd), "libnrnmech"), b"")
    return subprocess.CompletedProcess(args=cmd, returncode=0)


def test__neuron_version():
    with patch(
        "neurodamus.utils.compile_mods.importlib.metadata.version", return_value="8.2.6"
    ) as version:
        assert test_module._neuron_version() == "8.2.6"
        version.assert_called_once_with(test_module.NEURON_DISTRIBUTION)

    with patch(
        "neurodamus.utils.compile_mods.importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        assert test_module._neuron_version() is None


def test__toolchain_id(tmp_path):
    compiler_path = write(tmp_path / "nrnivmodl", b"#!/bin/sh\n")

    with patch("neurodamus.utils.compile_mods._neuron_version", return_value="9.0"):
        tid = test_module._toolchain_id(str(compiler_path))
    assert "neuron-9.0" in tid
    assert platform.machine() in tid
    assert compiler_path.name in tid
    assert str(compiler_path) not in tid

    other_compiler_path = write(tmp_path / "other" / "nrnivmodl", b"different wrapper")
    with patch("neurodamus.utils.compile_mods._neuron_version", return_value="9.0"):
        assert tid == test_module._toolchain_id(str(other_compiler_path))

    with patch("neurodamus.utils.compile_mods._neuron_version", return_value=None):
        tid = test_module._toolchain_id(str(compiler_path))
    assert "size-" in tid
    assert "mtime-" in tid

    with patch("neurodamus.utils.compile_mods._neuron_version", return_value=None):
        tid = test_module._toolchain_id(str(tmp_path / "does-not-exist"))
    assert "size-" not in tid


def test__resolve_nrnivmodl():
    assert test_module._resolve_nrnivmodl("/custom/nrnivmodl") == "/custom/nrnivmodl"

    with (
        patch("neurodamus.utils.compile_mods.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="nrnivmodl not found in PATH"),
    ):
        test_module._resolve_nrnivmodl(None)

    with patch(
        "neurodamus.utils.compile_mods.shutil.which", return_value="/a/neurodamus-nrnivmodl"
    ):
        assert test_module._resolve_nrnivmodl(None) == "/a/neurodamus-nrnivmodl"


def test__cache_key_varies_with_inputs(tmp_path):
    mod_dir = tmp_path / "mods"
    write(mod_dir / "a.mod", b"aaa")
    mod_files = test_module._get_mod_files([mod_dir])

    key = test_module._cache_key(mod_files, options())
    assert key == test_module._cache_key(mod_files, options())

    assert key != test_module._cache_key(mod_files, options(simulator=Simulator.coreneuron))
    assert key != test_module._cache_key(mod_files, options(incflags="-DX"))
    assert key != test_module._cache_key(mod_files, options(loadflags="-lfoo"))
    assert key != test_module._cache_key(mod_files, options(toolchain="toolchain-b"))

    write(mod_dir / "a.mod", b"bbb")
    assert key != test_module._cache_key(test_module._get_mod_files([mod_dir]), options())

    write(mod_dir / "b.mod", b"ccc")
    assert key != test_module._cache_key(test_module._get_mod_files([mod_dir]), options())


def test__cache_key_independent_of_input_dir(tmp_path):
    """Identical mod content in different directories must share a single cache slot."""
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    write(d1 / "a.mod", b"same contents")
    write(d2 / "a.mod", b"same contents")

    k1 = test_module._cache_key(test_module._get_mod_files([d1]), options())
    k2 = test_module._cache_key(test_module._get_mod_files([d2]), options())
    assert k1 == k2


def test__cache_key_independent_of_discovery_order(tmp_path):
    """The same set of mod files must hash identically however it was discovered.

    Otherwise `--input-dir a b` and `--input-dir b a` fragment the cache, as does any
    change in filesystem iteration order.
    """
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    write(d1 / "x.mod", b"xxx")
    write(d2 / "y.mod", b"yyy")
    write(d2 / "z.mod", b"zzz")

    forward = test_module._get_mod_files([d1, d2])
    reverse = test_module._get_mod_files([d2, d1])
    assert set(forward.values()) == set(reverse.values())
    assert test_module._cache_key(forward, options()) == test_module._cache_key(reverse, options())

    shuffled = dict(reversed(list(forward.items())))
    assert test_module._cache_key(shuffled, options()) == test_module._cache_key(forward, options())


def test__cache_slot(tmp_path):
    assert test_module._cache_slot(tmp_path, "abcdef1234") == tmp_path / "ab" / "cd" / "ef1234"


def test__slot_lock(tmp_path):
    slot = tmp_path / "slot"
    assert not test_module._slot_in_use(slot)

    with test_module._slot_lock(slot):
        assert (slot / test_module.LOCK_FILE).exists()
        assert test_module._slot_in_use(slot)

    assert not test_module._slot_in_use(slot)


def test__slot_lock_times_out(tmp_path, caplog):
    """A holder that hangs rather than dies must not block others forever."""
    slot = tmp_path / "slot"

    with test_module._slot_lock(slot):
        started = time.monotonic()
        with (
            caplog.at_level(logging.INFO),
            pytest.raises(RuntimeError) as excinfo,
            test_module._slot_lock(slot, timeout=0.3),
        ):
            pass
        waited = time.monotonic() - started

    assert 0.2 < waited < 10
    assert "Waiting for another build to release" in caplog.text
    message = str(excinfo.value)
    assert "Timed out after" in message
    assert "nrnivmodl" in message
    assert "--lock-timeout" in message

    assert not test_module._slot_in_use(slot)


def test__slot_lock_waits_then_acquires(tmp_path):
    """Waiting is not an error: the lock is taken as soon as the holder releases."""
    slot = tmp_path / "slot"
    released = threading.Event()

    def holder():
        with test_module._slot_lock(slot):
            time.sleep(0.4)
            released.set()

    thread = threading.Thread(target=holder)
    thread.start()
    time.sleep(0.1)
    with test_module._slot_lock(slot, timeout=30):
        assert released.is_set()
    thread.join()


def test__slot_lock_unsupported_filesystem(tmp_path, caplog):
    """Lustre without `-o flock` returns ENOLCK; the build must still run."""
    slot = tmp_path / "slot"
    with (
        caplog.at_level(logging.WARNING),
        patch(
            "neurodamus.utils.compile_mods.fcntl.flock",
            side_effect=OSError(errno.ENOLCK, "No locks available"),
        ),
        test_module._slot_lock(slot),
    ):
        pass
    assert "Locking is unsupported" in caplog.text


def test__acquire_lock_zero_timeout_waits_indefinitely(tmp_path):
    """`--lock-timeout 0` opts back in to blocking forever; it must still take a free lock."""
    slot = tmp_path / "slot"
    slot.mkdir()
    with open(slot / test_module.LOCK_FILE, "ab") as fd:
        assert test_module._acquire_lock(fd, slot / test_module.LOCK_FILE, 0) is True


def test__slot_in_use_unsupported_filesystem(tmp_path):
    """If we cannot test the lock, assume in use rather than clear a live build."""
    slot = tmp_path / "slot"
    slot.mkdir()
    (slot / test_module.LOCK_FILE).touch()
    with patch(
        "neurodamus.utils.compile_mods.fcntl.flock",
        side_effect=OSError(errno.ENOLCK, "No locks available"),
    ):
        assert test_module._slot_in_use(slot) is True


def test__touch_slot(tmp_path):
    slot = tmp_path / "slot"
    slot.mkdir()
    old_time = time.time() - 10_000
    os.utime(slot, (old_time, old_time))

    test_module._touch_slot(slot)
    assert slot.stat().st_mtime > old_time

    test_module._touch_slot(tmp_path / "missing")


def test__clear_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    fresh = test_module._cache_slot(cache_dir, "aabbfresh")
    stale = test_module._cache_slot(cache_dir, "ccddstale")
    locked = test_module._cache_slot(cache_dir, "eeffLocked")
    for slot in (fresh, stale, locked):
        write(slot / "modules.json", b"{}")

    old_time = time.time() - 60 * 86400
    os.utime(stale, (old_time, old_time))

    with test_module._slot_lock(locked):
        os.utime(locked, (old_time, old_time))
        removed = test_module._clear_cache(cache_dir, max_age_days=30)

    assert removed == [stale]
    assert fresh.exists()
    assert not stale.exists()
    assert locked.exists()


def test__clear_cache_empty(tmp_path):
    assert test_module._clear_cache(tmp_path / "does-not-exist", max_age_days=1) == []


def test__build_mod_files_cache_dir_hit(tmp_path):
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        first = test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)
        assert mock_run.call_count == 1

        second = test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)
        assert mock_run.call_count == 1

    assert first == second

    slot = test_module._cache_slot(
        cache_dir,
        test_module._cache_key({p: test_module._md5sum(p) for p in [input_dir / "a.mod"]}, opts),
    )
    assert first["NRNMECH_LIB_PATH"].startswith(str(slot))


def test__build_mod_files_cache_dir_miss_on_change(tmp_path):
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        test_module._build_mod_files([input_dir], None, "echo", options(), cache_dir)
        assert mock_run.call_count == 1

        test_module._build_mod_files(
            [input_dir], None, "echo", options(toolchain="toolchain-b"), cache_dir
        )
        assert mock_run.call_count == 2

        write(input_dir / "a.mod", b"other contents")
        test_module._build_mod_files([input_dir], None, "echo", options(), cache_dir)
        assert mock_run.call_count == 3

        write(input_dir / "a.mod", b"mod file")
        test_module._build_mod_files([input_dir], None, "echo", options(), cache_dir)
        assert mock_run.call_count == 3


def test__build_mod_files_concurrent(tmp_path):
    """Two builders racing on one cache slot must compile exactly once."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    calls = []

    def slow_nrnivmodl(cmd, cwd=None, **kwargs):
        calls.append(cwd)
        time.sleep(0.3)
        return fake_nrnivmodl(cmd, cwd=cwd, **kwargs)

    with (
        patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=slow_nrnivmodl),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [
            pool.submit(test_module._build_mod_files, [input_dir], None, "echo", opts, cache_dir)
            for _ in range(2)
        ]
        results = [f.result() for f in futures]

    assert len(calls) == 1
    assert results[0] == results[1]


def test__build_mod_files_failure_leaves_no_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    def failing(cmd, **_kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1)

    with (
        patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=failing),
        pytest.raises(RuntimeError, match="Failed to compile"),
    ):
        test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)

    mod = input_dir / "a.mod"
    slot = test_module._cache_slot(
        cache_dir, test_module._cache_key({mod: test_module._md5sum(mod)}, opts)
    )
    assert not test_module._metadata_path(slot).exists()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)
        assert mock_run.call_count == 1


def test__clear_cache_keeps_slot_being_rebuilt(tmp_path):
    """A rebuild of an already-stale slot must not be removed while it runs."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    mod = input_dir / "a.mod"
    slot = test_module._cache_slot(
        cache_dir, test_module._cache_key({mod: test_module._md5sum(mod)}, opts)
    )
    slot.mkdir(parents=True)
    old_time = time.time() - 60 * 86400
    os.utime(slot, (old_time, old_time))

    with patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl):
        test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)

    assert test_module._clear_cache(cache_dir, max_age_days=30) == []
    assert slot.exists()


def test__cache_key_different_filename_same_content(tmp_path):
    """Renaming a mod file changes the build, even though the bytes are unchanged."""
    mod_dir = tmp_path / "mods"
    write(mod_dir / "A.mod", b"same content")
    key_a = test_module._cache_key(test_module._get_mod_files([mod_dir]), options())

    (mod_dir / "A.mod").unlink()
    write(mod_dir / "B.mod", b"same content")
    key_b = test_module._cache_key(test_module._get_mod_files([mod_dir]), options())

    assert key_a != key_b


def test__cache_key_deterministic_across_processes(tmp_path):
    """The key must not depend on PYTHONHASHSEED: a shared cache spans processes.

    `_extract_mechanisms_dir` builds its input dirs from a `set`, whose iteration order
    varies per process, so the key has to be insensitive to the order it receives them.
    """
    names = ("alpha", "beta", "gamma", "delta")
    for i, name in enumerate(names):
        write(tmp_path / name / f"{name}.mod", f"mod {i}".encode())

    script = (
        "import sys;"
        "from pathlib import Path;"
        "from neurodamus.utils.compile_mods import _cache_key, _get_mod_files,"
        " Options, Simulator;"
        "o = Options(simulator=Simulator.neuron, incflags='', loadflags='', toolchain='t');"
        # list(set(...)) mirrors _extract_mechanisms_dir: order varies per process
        "dirs = list({Path(d) for d in sys.argv[1:]});"
        "print(_cache_key(_get_mod_files(dirs), o))"
    )
    keys = set()
    for seed in ("0", "1", "42", "1234"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO_ROOT)}
        out = subprocess.run(
            [sys.executable, "-c", script, *[str(tmp_path / n) for n in names]],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        keys.add(out.stdout.strip())

    assert len(keys) == 1, f"cache key varies with PYTHONHASHSEED: {keys}"


def test__build_mod_files_requires_mod_files(tmp_path):
    """Unlike a single-cell service, neurodamus always has mods to build: this is an error.

    Both a missing and an empty input directory are rejected before anything is compiled.
    """
    opts = options()
    with patch("neurodamus.utils.compile_mods.subprocess.run") as mock_run:
        with pytest.raises(RuntimeError, match="No mod files selected to be compiled"):
            test_module._build_mod_files(
                [tmp_path / "does-not-exist"], None, "echo", opts, tmp_path / "cache"
            )

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="No mod files selected to be compiled"):
            test_module._build_mod_files([empty], None, "echo", opts, tmp_path / "cache")

        mock_run.assert_not_called()

    assert not (tmp_path / "cache").exists()


def test__build_mod_files_shared_across_input_dirs(tmp_path):
    """A second consumer with the same mods reuses the slot instead of recompiling."""
    cache_dir = tmp_path / "cache"
    first_dir = tmp_path / "model-a" / "mechanisms"
    second_dir = tmp_path / "model-b" / "mechanisms"
    write(first_dir / "test.mod", b"NEURON { SUFFIX test }")
    write(second_dir / "test.mod", b"NEURON { SUFFIX test }")
    opts = options()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        first = test_module._build_mod_files([first_dir], None, "echo", opts, cache_dir)
        assert mock_run.call_count == 1

        second = test_module._build_mod_files([second_dir], None, "echo", opts, cache_dir)
        assert mock_run.call_count == 1

    assert first == second


def test__build_mod_files_stale_artifacts_do_not_shortcircuit(tmp_path):
    """Pre-existing compiled output must not be reused once the mod files change.

    There is deliberately no "output directory already exists, skip" shortcut: the
    decision is always made on content. With a fixed `--output-dir` the stale artifacts
    sit in the very directory being reused, so this is where such a shortcut would bite.
    """
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"original")
    opts = options()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        test_module._build_mod_files([input_dir], output_dir, "echo", opts)
        assert mock_run.call_count == 1

        assert test_module._metadata_path(output_dir).exists()
        assert test_module._get_dynamic_file(output_dir, "libnrnmech").exists()

        write(input_dir / "a.mod", b"edited")
        test_module._build_mod_files([input_dir], output_dir, "echo", opts)
        assert mock_run.call_count == 2

        test_module._build_mod_files([input_dir], output_dir, "echo", opts)
        assert mock_run.call_count == 2


def test__build_mod_files_cache_dir_keeps_both_builds(tmp_path):
    """Editing a mod file moves to a new slot without destroying the previous build."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"original")
    opts = options()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        built = test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)
        write(input_dir / "a.mod", b"edited")
        rebuilt = test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)
        assert mock_run.call_count == 2

    assert rebuilt["NRNMECH_LIB_PATH"] != built["NRNMECH_LIB_PATH"]
    assert Path(built["NRNMECH_LIB_PATH"]).exists()


def test__check_cache_rejects_removed_artifacts(tmp_path):
    """A slot whose library was deleted is a miss, not a hit onto a missing file."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        env = test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)
        assert mock_run.call_count == 1

        Path(env["NRNMECH_LIB_PATH"]).unlink()
        test_module._build_mod_files([input_dir], None, "echo", opts, cache_dir)
        assert mock_run.call_count == 2


def test__build_parser_destination():
    parser = test_module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "a", "--cache-dir", "b"])

    assert parser.parse_args(["--output-dir", "a"]).output_dir == "a"
    assert parser.parse_args(["--cache-dir", "b"]).cache_dir == "b"
    assert parser.parse_args(["--cache-dir", "b"]).max_age_days == test_module.DEFAULT_MAX_AGE_DAYS
    assert test_module.DEFAULT_LOCK_TIMEOUT_SECONDS == 300
    assert test_module.DEFAULT_MAX_AGE_DAYS == 30
    assert (
        parser.parse_args(["--cache-dir", "b"]).lock_timeout
        == test_module.DEFAULT_LOCK_TIMEOUT_SECONDS
    )
    assert parser.parse_args(["--cache-dir", "b", "--lock-timeout", "0"]).lock_timeout == 0


def test_compile_mods_print_cache_path(tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    cache_dir = tmp_path / "cache"

    argv = [
        "neurodamus-compile-mods",
        "--cache-dir",
        str(cache_dir),
        "--input-dir",
        str(input_dir),
        "--nrnivmodl",
        "/bin/echo",
        "--print-cache-path",
    ]
    with patch.object(sys, "argv", argv):
        test_module.compile_mods()

    out = capsys.readouterr().out.strip()
    assert out.startswith(str(cache_dir))
    assert not Path(out).exists()


def test_compile_mods_print_cache_path_no_mods(tmp_path):
    argv = [
        "neurodamus-compile-mods",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--nrnivmodl",
        "/bin/echo",
        "--print-cache-path",
    ]
    with (
        patch.object(sys, "argv", argv),
        pytest.raises(RuntimeError, match="No mod files selected to be compiled"),
    ):
        test_module.compile_mods()


def test_compile_mods_clear_cache_requires_cache_dir(tmp_path):
    argv = ["neurodamus-compile-mods", "--output-dir", str(tmp_path), "--clear-cache"]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        test_module.compile_mods()


def test_compile_mods_clear_cache(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    stale = test_module._cache_slot(cache_dir, "ccddstale")
    write(stale / "modules.json", b"{}")
    old_time = time.time() - 60 * 86400
    os.utime(stale, (old_time, old_time))

    argv = ["neurodamus-compile-mods", "--cache-dir", str(cache_dir), "--clear-cache"]
    with patch.object(sys, "argv", argv):
        test_module.compile_mods()

    assert not stale.exists()
    captured = capsys.readouterr()
    assert "Cleared 1 cache slot(s)" in captured.err
    assert not captured.out


def test_compile_mods_cache_dir_end_to_end(tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    cache_dir = tmp_path / "cache"

    argv = [
        "neurodamus-compile-mods",
        "--cache-dir",
        str(cache_dir),
        "--input-dir",
        str(input_dir),
        "--nrnivmodl",
        "/bin/echo",
        "--output-type",
        "shell",
    ]
    with patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl):
        with patch.object(sys, "argv", argv):
            test_module.compile_mods()
        first = capsys.readouterr()
        assert "NRNMECH_LIB_PATH=" in first.out
        assert "SPECIALS_PATH=" in first.out
        assert str(cache_dir) in first.out
        assert "Compiling 1 mod file(s)" in first.err

        with patch.object(sys, "argv", argv):
            test_module.compile_mods()
        second = capsys.readouterr()
        assert second.out == first.out
        assert "Cache hit" in second.err
        assert "Compiling" not in second.err
