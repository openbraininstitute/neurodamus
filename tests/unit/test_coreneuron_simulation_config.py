import pytest
from neurodamus.core.coreneuron_simulation_config import CoreSimulationConfig
from dataclasses import fields, MISSING
from typing import Union
from pathlib import Path

def make_minimal_kwargs():
    return {
        "outpath": "out",
        "datpath": "data",
        "tstop": 10.0,
        "dt": 0.1,
        "prcellgid": 42,
        "celsius": 37.0,
        "voltage": -65.0,
        "cell_permute": 2,
        "mpi": 1,
    }

def test_init_type_coercion():
    kwargs = make_minimal_kwargs()
    # Pass everything as string
    str_kwargs = {k: str(v) for k, v in kwargs.items()}
    cfg = CoreSimulationConfig(**str_kwargs)
    assert cfg.tstop == 10.0
    assert cfg.dt == 0.1
    assert cfg.prcellgid == 42
    assert cfg.celsius == 37.0
    assert cfg.voltage == -65.0
    assert cfg.cell_permute == 2
    assert cfg.mpi == 1

def test_optional_fields():
    kwargs = make_minimal_kwargs()
    kwargs["pattern"] = "sine"
    kwargs["seed"] = "123"
    kwargs["model_stats"] = "true"
    kwargs["report_conf"] = "conf.yml"
    cfg = CoreSimulationConfig(**kwargs)
    assert cfg.pattern == "sine"
    assert cfg.seed == 123
    assert cfg.model_stats is True
    assert cfg.report_conf == str(Path("conf.yml").resolve())

def test_missing_mandatory_raises():
    kwargs = make_minimal_kwargs()
    del kwargs["outpath"]
    with pytest.raises(TypeError):
        CoreSimulationConfig(**kwargs)

def test_dump_and_load(tmp_path):
    kwargs = make_minimal_kwargs()
    kwargs["model_stats"] = True
    kwargs["pattern"] = "sine"
    kwargs["report_conf"] = "conf.yml"
    cfg = CoreSimulationConfig(**kwargs)
    path = tmp_path / "config.txt"
    cfg.dump(path)
    assert path.exists()
    loaded_cfg = CoreSimulationConfig.load(path)
    assert cfg == loaded_cfg

def test_malformed_line_raises(tmp_path):
    path = tmp_path / "bad_config.txt"
    path.write_text("outpath='out'\ndatpath\n")
    with pytest.raises(ValueError):
        CoreSimulationConfig.load(path)

def test_string_quotes_handling(tmp_path):
    kwargs = make_minimal_kwargs()
    kwargs["pattern"] = "sine"
    cfg = CoreSimulationConfig(**kwargs)
    path = tmp_path / "config.txt"
    cfg.dump(path)
    # Manually check that quotes exist
    text = path.read_text()
    assert "pattern='sine'" in text
    # Load and check that quotes are stripped
    loaded_cfg = CoreSimulationConfig.load(path)
    assert loaded_cfg.pattern == "sine"

def test_mpi_as_int(tmp_path):
    kwargs = make_minimal_kwargs()
    kwargs["mpi"] = 0
    cfg = CoreSimulationConfig(**kwargs)
    path = tmp_path / "config.txt"
    cfg.dump(path)
    loaded_cfg = CoreSimulationConfig.load(path)
    assert isinstance(loaded_cfg.mpi, int)
    assert loaded_cfg.mpi == 0
