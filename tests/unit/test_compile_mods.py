import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import neurodamus.utils.compile_mods as test_module
import pytest

Options = test_module.Options
Simulator = test_module.Simulator


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

    # Create a pre-existing .mod file that should be cleaned
    mod_dir.mkdir(parents=True)
    (mod_dir / "old.mod").write_text("old")

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.mod").write_text("aaa")
    (src_dir / "b.mod").write_text("bbb")

    result = test_module._place_mod_files(
        output_dir, [src_dir / "a.mod", src_dir / "b.mod"]
    )

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
    write(d2 / "z.mod", b"content_y") # same content as `y.mod`

    with caplog.at_level(logging.WARNING):
        res = test_module._get_mod_files([d1, d2])

    assert "Already seen `x.mod`" in caplog.text
    assert "Already added a file with the same contents: 531ba6ac24a2ebde442987529f07e697" in caplog.text
    names = {p.name for p in res}
    assert "x.mod" in names
    assert "y.mod" in names
    assert next(p for p in res if p.name == "x.mod").parent == d2.absolute()


def test__generate_mod_metadata(tmp_path):
    path = write(tmp_path / "a.mod", b"mod content")

    mod_files = {path: "abc123"}
    options = Options(simulator=Simulator.neuron, incflags="-DFOO", loadflags="")

    meta = test_module._generate_mod_metadata(mod_files, options)
    assert meta["version"] == 1
    assert meta["hashes"] == [["a.mod", "abc123"]]
    assert meta["simulator"] == "neuron"
    assert meta["incflags"] == "-DFOO"


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

        # cache exists
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
