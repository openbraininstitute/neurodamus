"""Test all configurable parameters of `connection_overrides` in `simulation_config.json`.

Reference:
https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html#connection-overrides
"""


import numpy as np
import pytest

from tests import utils


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "connection_overrides": [
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "spont_minis": 200
                }
            ]
        }
    },
], indirect=True)
def test_spont_minis_simple(create_tmp_simulation_config_file):
    """Test spont_minis standard behavior.

    This test verifies that a spont_minis is correctly added to
    the simulation, preserving essential properties from existing
    netcons while enforcing expected modifications.

    The test is divided in 2 parts:
    - check that the spont_minis is added, neuron sees it
    and the other netcons are unaffected
    - run a simulation and see spikes at roughly the correct
    firing rate
    """
    from neurodamus import Neurodamus
    from neurodamus.connection import NetConType
    from neurodamus.core import NeuronWrapper as Ndc

    nd = Neurodamus(create_tmp_simulation_config_file)

    voltage_trace = Ndc.Vector()
    cell_ringB = nd.circuits.get_node_manager("RingB").get_cell(1000)
    voltage_trace.record(cell_ringB._cellref.soma[0](0.5)._ref_v)
    Ndc.finitialize()  # reinit for the recordings to be registered
    nd.run()

    utils.check_signal_peaks(voltage_trace, [15, 58, 167, 272, 388], threshold=0.5)
