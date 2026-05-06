import pytest
import numpy as np
from scipy.signal import find_peaks


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "inputs": {
                "Stimulus": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 5,
                    "duration": 50,
                    "node_set": "RingA",
                    "represents_physical_electrode": True,
                    "amp_start": 10,
                    "width": 1,
                    "frequency": 50
                }
            }
        }
    },
], indirect=True)
@pytest.mark.forked
def test_current_injection(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Nd

    nd = Neurodamus(create_tmp_simulation_config_file)

    cell_id = 1000
    manager = nd.circuits.get_node_manager("RingB")
    cell_ringB = manager.get_cell(cell_id)
    voltage_vec = Nd.Vector()
    voltage_vec.record(cell_ringB._cellref.soma[0](0.5)._ref_v)

    Nd.finitialize()
    nd.run()

    peaks_pos = find_peaks(voltage_vec, prominence=1)[0]
    np.testing.assert_allclose(peaks_pos, [92, 291])
