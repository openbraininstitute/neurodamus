import json
import os
import pytest
from pathlib import Path

SIM_DIR = Path(__file__).parent.absolute() / "simulations"
USECASE3 = SIM_DIR / "usecase3"


@pytest.fixture(scope="session")
def rootdir(request):
    return request.config.rootdir


@pytest.fixture(scope="session", name="SIM_DIR")
def sim_data_path():
    return SIM_DIR


@pytest.fixture(scope="session", name="USECASE3")
def usecase3_path():
    return USECASE3


@pytest.fixture
def sonata_config():
    return dict(
        manifest={"$CIRCUIT_DIR": str(USECASE3)},
        network="$CIRCUIT_DIR/circuit_config.json",
        node_sets_file="$CIRCUIT_DIR/nodesets.json",
        run={
            "random_seed": 12345,
            "dt": 0.05,
            "tstop": 10,
        }
    )


@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def create_tmp_simulation_file(request, tmp_path):
    """ copy simulation config file to tmp_path
    """
    params = request.param
    src_dir = Path(params.get("src_dir"))
    extra_config = params.get("extra_config")
    config_file = Path(params.get("simconfig_file"))
    with open(str(src_dir / config_file)) as src_f:
        sim_config_data = json.load(src_f)
    circuit_conf = sim_config_data.get("network", "circuit_config.json")
    if not os.path.isabs(circuit_conf):
        sim_config_data["network"] = str(src_dir / circuit_conf)
    node_sets_file = sim_config_data.get("node_sets_file")
    if node_sets_file and not os.path.isabs(node_sets_file):
        sim_config_data["node_sets_file"] = str(src_dir / node_sets_file)
    for input in sim_config_data.get("inputs", {}).values():
        spike_file = input.get("spike_file", "")
        if spike_file and not os.path.isabs(spike_file):
            input["spike_file"] = str(src_dir / input["spike_file"])
    if extra_config:
        sim_config_data.update(extra_config)
    with open(str(tmp_path / config_file), "w") as dst_f:
        json.dump(sim_config_data, dst_f, indent=2)
    return str(tmp_path / config_file)
