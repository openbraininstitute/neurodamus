import json
import pytest
from pathlib import Path
import platform

# utils needs to be registered to let pytest rewrite
# properly the assert errors. Either import it or use:
pytest.register_assert_rewrite("tests.utils")
#
# Example:
# a, b = 3, 4
# a == b
# this should say somewhere:
# AssertError 3 =/= 4


SIM_DIR = Path(__file__).parent.absolute() / "simulations"
USECASE3 = SIM_DIR / "usecase3"
PLATFORM_SYSTEM = platform.system()


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
    """change the working directory to tmp_path per test function automatically
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def create_tmp_simulation_config_file(request, tmp_path):
    """create simulation config file in tmp_path from
        1. simconfig_fixture in request: fixture's name (str)
        2. or simconfig_data in request: dict
        3. or copy of simconfig_file in request, and attach relative paths to src_dir
    Updates the config file with extra_config
    Returns the tmp file path
    """
    # import locally to register it in the pytests.
    # check the explanation about
    # pytest.register_assert_rewrite("tests.utils")
    # at the beginning of the file
    from tests import utils

    params = request.param
    src_dir = Path(params.get("src_dir", ""))
    config_file = Path(params.get("simconfig_file", "simulation_config.json"))
    sim_config_data = params.get("simconfig_data")

    if "simconfig_fixture" in params:
        sim_config_data = request.getfixturevalue(params.get("simconfig_fixture"))
    if not sim_config_data:
        # read sim_config_data from a config file
        with open(src_dir / config_file) as src_f:
            sim_config_data = json.load(src_f)

    if "extra_config" in params:
        sim_config_data = utils.merge_dicts(sim_config_data, params.get("extra_config"))

    # patch the relative file path to src_dir
    circuit_conf = sim_config_data.get("network", "circuit_config.json")
    if _is_valid_relative_path(circuit_conf):
        sim_config_data["network"] = str(src_dir / circuit_conf)
    node_sets_file = sim_config_data.get("node_sets_file", "")
    if _is_valid_relative_path(node_sets_file):
        sim_config_data["node_sets_file"] = str(src_dir / node_sets_file)
    for input in sim_config_data.get("inputs", {}).values():
        spike_file = input.get("spike_file", "")
        if _is_valid_relative_path(spike_file):
            input["spike_file"] = str(src_dir / input["spike_file"])

    with open(tmp_path / config_file, "w") as dst_f:
        json.dump(sim_config_data, dst_f, indent=2)
    return str(tmp_path / config_file)


def _is_valid_relative_path(filepath: str):
    return (
        filepath
        and not Path(filepath).is_absolute()
        and not Path(filepath).parts[0].startswith("$")
    )
