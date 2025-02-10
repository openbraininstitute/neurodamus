import pytest
from pathlib import Path
# !! NOTE: Please don't import NEURON/Neurodamus at module level
# pytest weird discovery system will trigger NEURON init and open a can of worms

# Make all tests run forked
pytestmark = pytest.mark.forked

RINGTEST_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"


@pytest.fixture
def ringtest_baseconfig():
    return dict(
        manifest={"$CIRCUIT_DIR": str(RINGTEST_DIR)},
        network="$CIRCUIT_DIR/circuit_config.json",
        node_sets_file="$CIRCUIT_DIR/nodesets.json",
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
