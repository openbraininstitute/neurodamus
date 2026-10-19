import argparse
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

import neurodamus.utils.compile_mods as test_module

Options = test_module.Options
Simulator = test_module.Simulator

REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(autouse=True)
def _reset_compile_mods_logger():
    """`compile_mods()` installs a non-propagating handler; undo that between tests."""
    yield
    test_module.L.handlers.clear()
    test_module.L.propagate = True
    test_module.L.setLevel(logging.NOTSET)


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
    assert f"Already seen `x.mod` ({d1.absolute() / 'x.mod'})" in caplog.text
    assert f"overriding with `{d2.absolute() / 'x.mod'}`" in caplog.text
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
    options = Options(simulator=Simulator.neuron, incflags="-DFOO", loadflags="", toolchain="")

    meta = test_module._generate_mod_metadata(mod_files, options)
    assert meta["version"] == test_module.VERSION
    assert meta["hashes"] == [["a.mod", "abc123"]]
    assert meta["simulator"] == "neuron"
    assert meta["incflags"] == "-DFOO"
    assert not meta["toolchain"]


def test_compile_mods_normalizes_omitted_flags(tmp_path):
    """Omitted CLI flags must not fork the cache key from explicit empty strings."""
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    captured = {}

    def capture_build(input_dirs, output_dir, nrnivmodl, options, cache_dir=None):
        captured["options"] = options
        return {"NRNMECH_LIB_PATH": str(tmp_path / "lib.so"), "SPECIALS_PATH": str(tmp_path)}

    argv = [
        "neurodamus-compile-mods",
        "--output-dir",
        str(tmp_path / "output"),
        "--input-dir",
        str(input_dir),
        "--nrnivmodl",
        "/bin/echo",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch("neurodamus.utils.compile_mods._build_mod_files", side_effect=capture_build),
    ):
        test_module.compile_mods()

    assert captured["options"].incflags == ""
    assert captured["options"].loadflags == ""


def test__is_cache_entry_valid_missing(tmp_path):
    options = Options(simulator=Simulator.neuron, incflags="", loadflags="", toolchain="")
    assert not test_module._is_cache_entry_valid({}, tmp_path, options)


def test__write_cache_and_is_cache_entry_valid(tmp_path):
    path = write(tmp_path / "a.mod", b"content")
    mod_files = {path: "hash1"}
    options0 = Options(simulator=Simulator.neuron, incflags="-DX", loadflags="", toolchain="")

    test_module._write_cache(mod_files, tmp_path, options0)
    assert test_module._metadata_path(tmp_path).exists()

    assert not test_module._is_cache_entry_valid(mod_files, tmp_path, options0)

    write(test_module._get_dynamic_file(tmp_path, "libnrnmech"), b"")
    assert test_module._is_cache_entry_valid(mod_files, tmp_path, options0)

    options1 = Options(simulator=Simulator.coreneuron, incflags="-DX", loadflags="", toolchain="")
    test_module._write_cache(mod_files, tmp_path, options1)
    assert not test_module._is_cache_entry_valid(mod_files, tmp_path, options1)

    options2 = Options(
        simulator=Simulator.neuron, incflags="-DX", loadflags="-different", toolchain=""
    )
    assert not test_module._is_cache_entry_valid(mod_files, tmp_path, options2)


def test__output_files(tmp_path):
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    arch = platform.machine()
    base = tmp_path / arch
    base.mkdir()
    write(base / f"libnrnmech{ext}", b"")

    options = Options(simulator=Simulator.neuron, incflags="", loadflags="", toolchain="")
    res = test_module._output_files(tmp_path, options)
    assert res["NRNMECH_LIB_PATH"] == str(base / f"libnrnmech{ext}")
    assert res["SPECIALS_PATH"] == str(base)
    assert "CORENEURONLIB" not in res

    write(base / f"libcorenrnmech{ext}", b"")
    options = Options(simulator=Simulator.coreneuron, incflags="", loadflags="", toolchain="")
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

    options = Options(simulator=Simulator.coreneuron, incflags="", loadflags="", toolchain="")
    res = test_module._output_files(tmp_path, options)
    assert "CORENEURONLIB" in res


def test__internal_mods_path():
    paths = test_module._internal_mods_path()
    assert len(paths) == 1
    assert paths[0].name == "mod"
    assert paths[0].exists()


def test__build_mod_filess(tmp_path):
    options = Options(simulator=Simulator.neuron, incflags="", loadflags="", toolchain="")
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
    if not res:
        # libsonata without mechanisms_dir support returns nothing.
        import libsonata

        cc = libsonata.CircuitConfig.from_file(circuit_config)
        name = next(iter(cc.node_populations))
        if not hasattr(cc.node_population_properties(name), "mechanisms_dir"):
            pytest.skip("installed libsonata has no mechanisms_dir")
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
    compiler_path.chmod(0o755)

    with patch("neurodamus.utils.compile_mods._neuron_version", return_value="9.0"):
        tid = test_module._toolchain_id(str(compiler_path))
    assert "neuron-9.0" in tid
    assert platform.machine() in tid
    assert compiler_path.name in tid
    assert str(compiler_path) not in tid

    other_compiler_path = write(tmp_path / "other" / "nrnivmodl", b"different wrapper")
    other_compiler_path.chmod(0o755)
    with patch("neurodamus.utils.compile_mods._neuron_version", return_value="9.0"):
        assert tid == test_module._toolchain_id(str(other_compiler_path))

    with patch("neurodamus.utils.compile_mods._neuron_version", return_value=None):
        tid = test_module._toolchain_id(str(compiler_path))
    assert "size-" in tid
    assert "mtime-" in tid

    with (
        patch("neurodamus.utils.compile_mods._neuron_version", return_value=None),
        pytest.raises(OSError),
    ):
        test_module._toolchain_id(str(tmp_path / "does-not-exist"))


def test__resolve_nrnivmodl(tmp_path):
    compiler = write(tmp_path / "nrnivmodl", b"#!/bin/sh\n")
    compiler.chmod(0o755)
    assert test_module._resolve_nrnivmodl(str(compiler)) == str(compiler.resolve())

    with pytest.raises(RuntimeError, match="nrnivmodl not found:"):
        test_module._resolve_nrnivmodl(str(tmp_path / "missing"))

    not_exec = write(tmp_path / "not-exec", b"#!/bin/sh\n")
    not_exec.chmod(0o644)
    with pytest.raises(RuntimeError, match="nrnivmodl is not executable:"):
        test_module._resolve_nrnivmodl(str(not_exec))

    with (
        patch("neurodamus.utils.compile_mods.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="nrnivmodl not found in PATH"),
    ):
        test_module._resolve_nrnivmodl(None)

    with patch("neurodamus.utils.compile_mods.shutil.which", return_value=str(compiler)):
        assert test_module._resolve_nrnivmodl(None) == str(compiler.resolve())


def test__cache_entry_key_varies_with_inputs(tmp_path):
    mod_dir = tmp_path / "mods"
    write(mod_dir / "a.mod", b"aaa")
    mod_files = test_module._get_mod_files([mod_dir])

    key = test_module._cache_entry_key(mod_files, options())
    assert key == test_module._cache_entry_key(mod_files, options())

    assert key != test_module._cache_entry_key(mod_files, options(simulator=Simulator.coreneuron))
    assert key != test_module._cache_entry_key(mod_files, options(incflags="-DX"))
    assert key != test_module._cache_entry_key(mod_files, options(loadflags="-lfoo"))
    assert key != test_module._cache_entry_key(mod_files, options(toolchain="toolchain-b"))

    write(mod_dir / "a.mod", b"bbb")
    assert key != test_module._cache_entry_key(test_module._get_mod_files([mod_dir]), options())

    write(mod_dir / "b.mod", b"ccc")
    assert key != test_module._cache_entry_key(test_module._get_mod_files([mod_dir]), options())


def test__cache_entry_key_independent_of_input_dir(tmp_path):
    """Identical mod content in different directories must share a single cache entry."""
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    write(d1 / "a.mod", b"same contents")
    write(d2 / "a.mod", b"same contents")

    k1 = test_module._cache_entry_key(test_module._get_mod_files([d1]), options())
    k2 = test_module._cache_entry_key(test_module._get_mod_files([d2]), options())
    assert k1 == k2


def test__cache_entry_key_independent_of_discovery_order(tmp_path):
    """Hash a file set independently of discovery order."""
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    write(d1 / "x.mod", b"xxx")
    write(d2 / "y.mod", b"yyy")
    write(d2 / "z.mod", b"zzz")

    forward = test_module._get_mod_files([d1, d2])
    reverse = test_module._get_mod_files([d2, d1])
    assert set(forward.values()) == set(reverse.values())
    assert test_module._cache_entry_key(forward, options()) == test_module._cache_entry_key(
        reverse, options()
    )

    shuffled = dict(reversed(list(forward.items())))
    assert test_module._cache_entry_key(shuffled, options()) == test_module._cache_entry_key(
        forward, options()
    )


def test__cache_entry_path(tmp_path):
    assert test_module._cache_entry_path(tmp_path, "abcdef1234") == (
        tmp_path / "ab" / "cd" / "ef1234"
    )


def test__staging_and_to_delete_dirs_are_beside_entry(tmp_path):
    entry = tmp_path / "cache" / "ab" / "cd" / "rest"
    staging = test_module._staging_dir(entry)
    to_delete = test_module._to_delete_dir(entry)

    assert staging.parent == entry.parent == to_delete.parent
    assert staging.name.startswith(test_module._STAGING_PREFIX + "rest-")
    assert to_delete.name.startswith(test_module._TO_DELETE_PREFIX + "rest-")
    assert staging != entry != to_delete


def test__publish_cache_entry(tmp_path):
    entry = tmp_path / "ab" / "cd" / "rest"
    staging = test_module._staging_dir(entry)
    write(staging / "modules.json", b'{"ok": true}')

    assert test_module._publish_cache_entry(staging, entry) is True
    assert entry.exists()
    assert not staging.exists()
    assert (entry / "modules.json").read_bytes() == b'{"ok": true}'


def test__publish_cache_entry_loses_race_leaves_staging(tmp_path):
    entry = tmp_path / "ab" / "cd" / "rest"
    write(entry / "modules.json", b'{"winner": true}')
    staging = test_module._staging_dir(entry)
    write(staging / "modules.json", b'{"loser": true}')

    assert test_module._publish_cache_entry(staging, entry) is False
    assert staging.exists()
    assert (entry / "modules.json").read_bytes() == b'{"winner": true}'


def test__remove_cache_dir_unpublishes_via_rename(tmp_path):
    entry = tmp_path / "ab" / "cd" / "rest"
    write(entry / "modules.json", b"{}")

    renamed = []
    real_rename = Path.rename

    def tracking_rename(self, target):
        renamed.append((self, Path(target)))
        return real_rename(self, target)

    with patch.object(Path, "rename", tracking_rename):
        assert test_module._remove_cache_dir(entry) is True

    assert not entry.exists()
    assert list(entry.parent.glob(f"{test_module._TO_DELETE_PREFIX}*")) == []
    assert len(renamed) == 1
    assert renamed[0][0] == entry
    assert renamed[0][1].name.startswith(test_module._TO_DELETE_PREFIX + "rest-")


def test__remove_cache_dir_missing_path(tmp_path):
    assert test_module._remove_cache_dir(tmp_path / "missing") is False


def test__touch_cache_entry(tmp_path):
    entry = tmp_path / "entry"
    entry.mkdir()
    old_time = time.time() - 10_000
    os.utime(entry, (old_time, old_time))

    test_module._touch_cache_entry(entry)
    assert entry.stat().st_mtime > old_time

    test_module._touch_cache_entry(tmp_path / "missing")



def test__build_mod_files_compiles_in_staging_then_publishes(tmp_path):
    """A cache miss must never compile directly into the final entry path."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()
    entry = test_module._cache_entry_path(
        cache_dir, test_module._cache_entry_key(test_module._get_mod_files([input_dir]), opts)
    )

    seen = []

    def recording_nrnivmodl(cmd, cwd=None, **kwargs):
        cwd_path = Path(cwd)
        seen.append(cwd_path)
        assert cwd_path != entry
        assert cwd_path.name.startswith(test_module._STAGING_PREFIX)
        assert not entry.exists()
        return fake_nrnivmodl(cmd, cwd=cwd, **kwargs)

    with patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=recording_nrnivmodl):
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", opts, cache_dir)

    assert len(seen) == 1
    assert entry.exists()
    assert not seen[0].exists()
    assert list(entry.parent.glob(f"{test_module._STAGING_PREFIX}*")) == []
    assert test_module._is_cache_entry_valid(test_module._get_mod_files([input_dir]), entry, opts)


def test__build_mod_files_cache_hit_concurrent_readers(tmp_path):
    """Concurrent cache hits must both materialize without recompiling."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    with patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl):
        test_module._build_mod_files([input_dir], tmp_path / "first", "echo", opts, cache_dir)

    with (
        patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl) as run,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [
            pool.submit(
                test_module._build_mod_files,
                [input_dir],
                tmp_path / f"out-{i}",
                "echo",
                opts,
                cache_dir,
            )
            for i in range(2)
        ]
        results = [f.result() for f in futures]
        run.assert_not_called()

    assert all(r["NRNMECH_LIB_PATH"].startswith(str(tmp_path / "out-")) for r in results)


def test__build_mod_files_cache_dir_hit(tmp_path):
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        first = test_module._build_mod_files(
            [input_dir], tmp_path / "output", "echo", opts, cache_dir
        )
        assert mock_run.call_count == 1

        second = test_module._build_mod_files(
            [input_dir], tmp_path / "output", "echo", opts, cache_dir
        )
        assert mock_run.call_count == 1

    assert first == second
    assert first["NRNMECH_LIB_PATH"].startswith(str(tmp_path / "output"))

    cache_entry = test_module._cache_entry_path(
        cache_dir,
        test_module._cache_entry_key(
            {p: test_module._md5sum(p) for p in [input_dir / "a.mod"]}, opts
        ),
    )
    assert test_module._get_dynamic_file(cache_entry, "libnrnmech").exists()


def test__build_mod_files_cache_dir_miss_on_change(tmp_path):
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", options(), cache_dir)
        assert mock_run.call_count == 1

        test_module._build_mod_files(
            [input_dir], tmp_path / "output", "echo", options(toolchain="toolchain-b"), cache_dir
        )
        assert mock_run.call_count == 2

        write(input_dir / "a.mod", b"other contents")
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", options(), cache_dir)
        assert mock_run.call_count == 3

        write(input_dir / "a.mod", b"mod file")
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", options(), cache_dir)
        assert mock_run.call_count == 3


def test__build_mod_files_concurrent(tmp_path):
    """Racing builders may both compile; publish is atomic and both succeed."""
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
            pool.submit(
                test_module._build_mod_files,
                [input_dir],
                tmp_path / f"output-{i}",
                "echo",
                opts,
                cache_dir,
            )
            for i in range(2)
        ]
        results = [f.result() for f in futures]

    assert 1 <= len(calls) <= 2
    assert all(Path(cwd).name.startswith(test_module._STAGING_PREFIX) for cwd in calls)
    assert results[0]["NRNMECH_LIB_PATH"] != results[1]["NRNMECH_LIB_PATH"]
    entry = test_module._cache_entry_path(
        cache_dir, test_module._cache_entry_key(test_module._get_mod_files([input_dir]), opts)
    )
    assert test_module._is_cache_entry_valid(test_module._get_mod_files([input_dir]), entry, opts)
    assert list(entry.parent.glob(f"{test_module._STAGING_PREFIX}*")) == []


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
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", opts, cache_dir)

    mod = input_dir / "a.mod"
    entry = test_module._cache_entry_path(
        cache_dir, test_module._cache_entry_key({mod: test_module._md5sum(mod)}, opts)
    )
    assert not entry.exists()
    assert list(entry.parent.glob(f"{test_module._STAGING_PREFIX}*")) == []

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", opts, cache_dir)
        assert mock_run.call_count == 1


def test__compile_into_cache_replaces_invalid_entry(tmp_path):
    """An incomplete final path is unpublished so rename can publish a good build."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    mod = input_dir / "a.mod"
    entry = test_module._cache_entry_path(
        cache_dir, test_module._cache_entry_key({mod: test_module._md5sum(mod)}, opts)
    )
    entry.mkdir(parents=True)
    write(entry / "junk.txt", b"incomplete")

    with patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl):
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", opts, cache_dir)

    assert test_module._is_cache_entry_valid(test_module._get_mod_files([input_dir]), entry, opts)
    assert not (entry / "junk.txt").exists()


def test__compile_into_cache_retries_publish_after_failed_initial_remove(tmp_path):
    """If the first unpublish fails, publish retries once after a later successful remove."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()
    mod_files = test_module._get_mod_files([input_dir])
    entry = test_module._cache_entry_path(cache_dir, test_module._cache_entry_key(mod_files, opts))
    entry.mkdir(parents=True)
    write(entry / "junk.txt", b"incomplete")

    real_remove = test_module._remove_cache_dir
    calls = {"n": 0}

    def flaky_remove(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return real_remove(path)

    with (
        patch("neurodamus.utils.compile_mods._remove_cache_dir", side_effect=flaky_remove),
        patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl),
    ):
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", opts, cache_dir)

    assert calls["n"] == 2
    assert test_module._is_cache_entry_valid(mod_files, entry, opts)


def test__compile_into_cache_errors_if_invalid_leftover_cannot_be_removed(tmp_path):
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()
    mod_files = test_module._get_mod_files([input_dir])
    entry = test_module._cache_entry_path(cache_dir, test_module._cache_entry_key(mod_files, opts))
    entry.mkdir(parents=True)
    write(entry / "junk.txt", b"incomplete")

    with (
        patch("neurodamus.utils.compile_mods._remove_cache_dir", return_value=False),
        patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl),
        pytest.raises(RuntimeError, match="could not be removed"),
    ):
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", opts, cache_dir)

    assert entry.exists()
    assert list(entry.parent.glob(f"{test_module._STAGING_PREFIX}*")) == []


def test__is_cache_entry_valid_corrupt_metadata(tmp_path):
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()
    mod_files = test_module._get_mod_files([input_dir])
    entry = test_module._cache_entry_path(cache_dir, test_module._cache_entry_key(mod_files, opts))

    with patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl) as run:
        test_module._build_mod_files([input_dir], tmp_path / "first", "echo", opts, cache_dir)
        assert run.call_count == 1

        write(entry / "modules.json", b"not-json{")
        assert not test_module._is_cache_entry_valid(mod_files, entry, opts)

        test_module._build_mod_files([input_dir], tmp_path / "second", "echo", opts, cache_dir)
        assert run.call_count == 2

    assert test_module._is_cache_entry_valid(mod_files, entry, opts)


def test__materialize_rebuilds_when_entry_disappears_during_copy(tmp_path, caplog):
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    with patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl):
        test_module._build_mod_files([input_dir], tmp_path / "first", "echo", opts, cache_dir)

    real_copy = test_module._copy_cache_entry
    calls = {"n": 0}

    def copy_once_missing(cache_entry, output_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            shutil.rmtree(cache_entry)
            raise FileNotFoundError(cache_entry)
        return real_copy(cache_entry, output_dir)

    with (
        caplog.at_level(logging.WARNING),
        patch("neurodamus.utils.compile_mods._copy_cache_entry", side_effect=copy_once_missing),
        patch("neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl) as run,
    ):
        env = test_module._build_mod_files(
            [input_dir], tmp_path / "second", "echo", opts, cache_dir
        )

    assert calls["n"] == 2
    assert run.call_count == 1
    assert "disappeared while copying" in caplog.text
    assert Path(env["NRNMECH_LIB_PATH"]).exists()


def test__discard_staging_logs_at_debug(tmp_path, caplog):
    entry = tmp_path / "entry"
    staging = tmp_path / f"{test_module._STAGING_PREFIX}entry-1-deadbeef"
    staging.mkdir()
    write(staging / "modules.json", b"{}")

    with caplog.at_level(logging.DEBUG, logger=test_module.L.name):
        test_module._discard_staging(staging, entry)

    assert not staging.exists()
    assert "already published, discarding staging" in caplog.text


def test__cache_entry_key_different_filename_same_content(tmp_path):
    """Renaming a mod file changes the build, even though the bytes are unchanged."""
    mod_dir = tmp_path / "mods"
    write(mod_dir / "A.mod", b"same content")
    key_a = test_module._cache_entry_key(test_module._get_mod_files([mod_dir]), options())

    (mod_dir / "A.mod").unlink()
    write(mod_dir / "B.mod", b"same content")
    key_b = test_module._cache_entry_key(test_module._get_mod_files([mod_dir]), options())

    assert key_a != key_b


def test__cache_entry_key_deterministic_across_processes(tmp_path):
    """Keep cache keys stable across process hash seeds."""
    names = ("alpha", "beta", "gamma", "delta")
    for i, name in enumerate(names):
        write(tmp_path / name / f"{name}.mod", f"mod {i}".encode())

    script = (
        "import sys;"
        "from pathlib import Path;"
        "from neurodamus.utils.compile_mods import _cache_entry_key, _get_mod_files,"
        " Options, Simulator;"
        "o = Options(simulator=Simulator.neuron, incflags='', loadflags='', toolchain='t');"
        # list(set(...)) mirrors _extract_mechanisms_dir: order varies per process
        "dirs = list({Path(d) for d in sys.argv[1:]});"
        "print(_cache_entry_key(_get_mod_files(dirs), o))"
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
    """Reject missing or empty MOD inputs before compiling."""
    opts = options()
    with patch("neurodamus.utils.compile_mods.subprocess.run") as mock_run:
        with pytest.raises(RuntimeError, match="No mod files selected to be compiled"):
            test_module._build_mod_files(
                [tmp_path / "does-not-exist"], tmp_path / "output", "echo", opts, tmp_path / "cache"
            )

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="No mod files selected to be compiled"):
            test_module._build_mod_files(
                [empty], tmp_path / "output", "echo", opts, tmp_path / "cache"
            )

        mock_run.assert_not_called()

    assert not (tmp_path / "cache").exists()


def test__build_mod_files_shared_across_input_dirs(tmp_path):
    """A second consumer with the same mods reuses the entry instead of recompiling."""
    cache_dir = tmp_path / "cache"
    first_dir = tmp_path / "model-a" / "mechanisms"
    second_dir = tmp_path / "model-b" / "mechanisms"
    write(first_dir / "test.mod", b"NEURON { SUFFIX test }")
    write(second_dir / "test.mod", b"NEURON { SUFFIX test }")
    opts = options()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        first = test_module._build_mod_files(
            [first_dir], tmp_path / "output", "echo", opts, cache_dir
        )
        assert mock_run.call_count == 1

        second = test_module._build_mod_files(
            [second_dir], tmp_path / "output", "echo", opts, cache_dir
        )
        assert mock_run.call_count == 1

    assert first == second


def test__build_mod_files_stale_artifacts_do_not_shortcircuit(tmp_path):
    """Reject stale output after MOD content changes."""
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
    """Editing a mod file moves to a new entry without destroying the previous build."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"original")
    opts = options()

    original_entry = test_module._cache_entry_path(
        cache_dir, test_module._cache_entry_key(test_module._get_mod_files([input_dir]), opts)
    )
    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        built = test_module._build_mod_files(
            [input_dir], tmp_path / "output", "echo", opts, cache_dir
        )
        write(input_dir / "a.mod", b"edited")
        edited_entry = test_module._cache_entry_path(
            cache_dir, test_module._cache_entry_key(test_module._get_mod_files([input_dir]), opts)
        )
        rebuilt = test_module._build_mod_files(
            [input_dir], tmp_path / "output", "echo", opts, cache_dir
        )
        assert mock_run.call_count == 2

    assert rebuilt == built
    assert test_module._get_dynamic_file(original_entry, "libnrnmech").exists()
    assert test_module._get_dynamic_file(edited_entry, "libnrnmech").exists()


def test__cache_hit_rematerializes_removed_output(tmp_path):
    """A cache hit restores missing files in the output directory."""
    cache_dir = tmp_path / "cache"
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    opts = options()

    with patch(
        "neurodamus.utils.compile_mods.subprocess.run", side_effect=fake_nrnivmodl
    ) as mock_run:
        env = test_module._build_mod_files(
            [input_dir], tmp_path / "output", "echo", opts, cache_dir
        )
        assert mock_run.call_count == 1

        Path(env["NRNMECH_LIB_PATH"]).unlink()
        test_module._build_mod_files([input_dir], tmp_path / "output", "echo", opts, cache_dir)
        assert mock_run.call_count == 1
        assert Path(env["NRNMECH_LIB_PATH"]).exists()


def test__build_parser_destination():
    parser = test_module.build_parser()

    args = parser.parse_args(["--output-dir", "a", "--cache-dir", "b"])
    assert args.output_dir == "a"
    assert args.cache_dir == "b"

    assert parser.parse_args(["--output-dir", "a"]).output_dir == "a"
    assert not hasattr(parser.parse_args(["--output-dir", "a"]), "clear_cache")
    assert not hasattr(parser.parse_args(["--output-dir", "a"]), "lock_timeout")


def test_compile_mods_requires_output_dir():
    argv = ["neurodamus-compile-mods", "--nrnivmodl", "/bin/echo"]
    with (
        patch.object(sys, "argv", argv),
        patch("neurodamus.utils.compile_mods._build_mod_files") as build,
        pytest.raises(SystemExit),
    ):
        test_module.compile_mods()
    build.assert_not_called()


def test_compile_mods_cache_dir_end_to_end(tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")
    cache_dir = tmp_path / "cache"

    argv = [
        "neurodamus-compile-mods",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(tmp_path / "output"),
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
        assert str(tmp_path / "output") in first.out
        assert "Compiling 1 mod file(s)" in first.err

        with patch.object(sys, "argv", argv):
            test_module.compile_mods()
        second = capsys.readouterr()
        assert second.out == first.out
        assert "Cache hit" in second.err
        assert "Compiling" not in second.err
