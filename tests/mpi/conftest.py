import pytest

from pathlib import Path
# !! NOTE: Please don't import NEURON/Neurodamus at module level
# pytest weird discovery system will trigger NEURON init and open a can of worms

# Make all tests run forked
pytestmark = pytest.mark.forked

RINGTEST_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
NGV_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ngv"


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
