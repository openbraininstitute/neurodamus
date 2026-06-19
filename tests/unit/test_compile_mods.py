import platform
import sys

import neurodamus.utils.compile_mods as test_module

Options = test_module.Options
Simulator = test_module.Simulator


def write(path, contents):
    path.write_bytes(contents)
    return path


def test__metadata_path(tmp_path):
    assert test_module._metadata_path(tmp_path) == tmp_path / "modules.json"


def test__md5sum(tmp_path):
    path = write(tmp_path / "a.txt", b"hello")
    assert test_module._md5sum(path) == "5d41402abc4b2a76b9719d911017c592"


def test__get_mod_files(tmp_path):
    d1 = tmp_path / "a"
    d1.mkdir()
    d2 = tmp_path / "b"
    d2.mkdir()

    write(d1 / "x.mod", b"content_x")
    write(d2 / "y.mod", b"content_y")
    # same name in d2 overrides d1
    write(d2 / "x.mod", b"content_x_override")

    res = test_module._get_mod_files([d1, d2])
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
    options = Options(simulator=Simulator.coreneuron, incflags="-DX", loadflags="")

    test_module._write_cache(mod_files, tmp_path, options)
    assert test_module._metadata_path(tmp_path).exists()

    assert test_module._check_cache(mod_files, tmp_path, options)

    options2 = Options(simulator=Simulator.neuron, incflags="-DX", loadflags="")
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


#def test__build_modules(tmp_path):
#    options = Options(simulator=Simulator.coreneuron, incflags="", loadflags="")
#    res = test_module._build_modules(tmp_path, "echo", options)
