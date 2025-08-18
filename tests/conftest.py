import json
import pytest
import platform
from pathlib import Path

from neurodamus.core._utils import run_only_rank0

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI = False
    comm = None
    rank = 0  # fallback to single-process logic

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
V5_SONATA = SIM_DIR / "v5_sonata"
PLATFORM_SYSTEM = platform.system()
RINGTEST_DIR = Path(__file__).parent.absolute() / "simulations" / "ringtest"
NGV_DIR = Path(__file__).parent.absolute() / "simulations" / "ngv"


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
        },
        conditions={
            "v_init": -65,
        }
    )


@pytest.fixture
def ringtest_baseconfig():
    return dict(
        network=str(RINGTEST_DIR / "circuit_config.json"),
        node_sets_file=str(RINGTEST_DIR / "nodesets.json"),
        target_simulator="NEURON",
        run={
            "random_seed": 1122,
            "dt": 0.1,
            "tstop": 50,
        },
        node_set="Mosaic",
        conditions={
            "celsius": 35,
            "v_init": -65
        }
    )

@pytest.fixture
def v5_sonata_config():
    config_path = V5_SONATA / "simulation_config_mini.json"
    with open(config_path, "r") as f:
        d = json.load(f)
    d["compartment_sets_file"] = str(V5_SONATA / "compartment_sets.json")
    d["network"] = str(V5_SONATA / "sub_mini5" / "circuit_config.json")
    return d


@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch, tmp_path):
    """change the working directory to tmp_path per test function automatically
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def copy_memory_files(change_test_dir):
    # Fix values to ensure allocation memory (0,0)[1, 3] (1,0)[2]
    metypes_memory = {
        "MTYPE0-ETYPE0": 1000.0,
        "MTYPE0-ETYPE1": 100.0,
        "MTYPE1-ETYPE1": 200.0,
        "MTYPE2-ETYPE2": 200.0,
    }
    with Path("cell_memory_usage.json").open("w") as f:
        json.dump(metypes_memory, f, indent=4)


@run_only_rank0
def _create_simulation_config_file(params, dst_dir, sim_config_data=None) -> str:
    """create simulation config file in dst_dir from
        1. simconfig_data: dict
        2. or copy of simconfig_file in params, and attach relative paths to src_dir
    Updates the config file with extra_config
    Returns the file path
    """
    from tests import utils

    src_dir = Path(params.get("src_dir", ""))
    config_file = Path(params.get("simconfig_file", "simulation_config.json"))

    if not sim_config_data:
        sim_config_data = params.get("simconfig_data")
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

    with open(dst_dir / config_file, "w") as dst_f:
        json.dump(sim_config_data, dst_f, indent=2)
    return str(dst_dir / config_file)


def _is_valid_relative_path(filepath: str):
    return (
        filepath
        and not Path(filepath).is_absolute()
        and not Path(filepath).parts[0].startswith("$")
    )


@pytest.fixture
def create_tmp_simulation_config_file(request):
    """create simulation config file in tmp_path from
        1. simconfig_fixture in request: fixture's name (str)
        2. or simconfig_data in request: dict
        3. or copy of simconfig_file in request, and attach relative paths to src_dir
    Updates the config file with extra_config
    Returns the tmp file path
    """
    try:
        tmp_path = request.getfixturevalue("mpi_tmp_path")
    except pytest.FixtureLookupError:
        tmp_path = request.getfixturevalue("tmp_path")
    # import locally to register it in the pytests.
    # check the explanation about
    # pytest.register_assert_rewrite("tests.utils")
    # at the beginning of the file
    params = request.param
    sim_config_data = None
    if "simconfig_fixture" in params:
        sim_config_data = request.getfixturevalue(params.get("simconfig_fixture"))
    return _create_simulation_config_file(params, tmp_path, sim_config_data)


@pytest.fixture
def create_simulation_config_file_factory():
    """Returns the factory to allow the creation of the simulation config
    file
    """
    return _create_simulation_config_file
