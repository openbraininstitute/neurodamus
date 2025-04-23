import json
import pytest
from pathlib import Path
import platform
from neurodamus.core._utils import run_only_rank0

from tests import utils

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI = False
    comm = None
    rank = 0  # fallback to single-process logic

# Make all tests run forked
pytestmark = pytest.mark.forked

RINGTEST_DIR = Path(__file__).parent.absolute() / "simulations" / "ringtest"
NGV_DIR = Path(__file__).parent.absolute() / "simulations" / "ngv"
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
def create_tmp_simulation_config_file(request):
    """
    Creates a simulation config file in a temporary directory.

    Uses mpi_tmp_path if available, otherwise tmp_path.
    Builds the config from:
    1. A fixture name (simconfig_fixture)
    2. A data dict (simconfig_data)
    3. An existing file (simconfig_file), with paths adjusted to src_dir

    Applies extra_config overrides.
    Only rank 0 writes the file; others wait. Returns the config path.
    """
    try:
        tmp_path = request.getfixturevalue("mpi_tmp_path")
    except pytest.FixtureLookupError:
        tmp_path = request.getfixturevalue("tmp_path")

    params = request.param
    src_dir = Path(params.get("src_dir", ""))
    config_file = Path(params.get("simconfig_file", "simulation_config.json"))
    sim_config_path = tmp_path / config_file

    @run_only_rank0  # Ensure this block only executes on rank 0
    def write_config():
        sim_config_data = params.get("simconfig_data")

        if "simconfig_fixture" in params:
            sim_config_data = request.getfixturevalue(params.get("simconfig_fixture"))
        if not sim_config_data:
            with open(src_dir / config_file) as src_f:
                sim_config_data = json.load(src_f)

        if "extra_config" in params:
            sim_config_data = utils.merge_dicts(sim_config_data, params.get("extra_config"))

        # patch relative paths
        circuit_conf = sim_config_data.get("network", "circuit_config.json")
        if _is_valid_relative_path(circuit_conf):
            sim_config_data["network"] = str(src_dir / circuit_conf)
        node_sets_file = sim_config_data.get("node_sets_file", "")
        if _is_valid_relative_path(node_sets_file):
            sim_config_data["node_sets_file"] = str(src_dir / node_sets_file)
        for input in sim_config_data.get("inputs", {}).values():
            spike_file = input.get("spike_file", "")
            if _is_valid_relative_path(spike_file):
                input["spike_file"] = str(src_dir / spike_file)

        with open(sim_config_path, "w") as dst_f:
            json.dump(sim_config_data, dst_f, indent=2)

    # Call the write_config function that only runs on rank 0
    write_config()

    # Ensure all ranks wait for the file to be written
    if MPI:
        MPI.COMM_WORLD.Barrier()

    return str(sim_config_path)


def _is_valid_relative_path(filepath: str):
    return (
        filepath
        and not Path(filepath).is_absolute()
        and not Path(filepath).parts[0].startswith("$")
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
