import numpy as np
import pytest


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
    from neurodamus.core import NeurodamusCore as Nd

    nd = Neurodamus(create_tmp_simulation_config_file)

    cell_id = 1001
    manager = nd.circuits.get_node_manager("RingB")
    cell_ringB = manager.get_cell(cell_id)
    voltage_vec = Nd.Vector()
    voltage_vec.record(cell_ringB._cellref.soma[0](0.5)._ref_v)

    Nd.finitialize()
    nd.run()

    # Calculate the second-order difference of the 
    # voltage vector (v_increase_rate)
    v_increase_rate = np.diff(voltage_vec, 2)  
    # Convolve the v_increase_rate with a smoothing kernel 
    # [1, 2, 4, 2, 1] to reduce noise
    window_sum = np.convolve(v_increase_rate, [1, 2, 4, 2, 1], 'valid')  
    # Find the positions where the window sum is below -0.5, 
    # indicating the beginning of a peak in voltage_vec
    strong_reduction_pos = np.nonzero(window_sum < -0.5)[0]  
    # Filter out consecutive positions, the negative second
    # derivative persists for a while
    peaks_pos = strong_reduction_pos[np.insert(
        np.diff(strong_reduction_pos) > 1, 0, True)]  
    # Define expected positions where peaks are expected
    expected_peaks_pos = np.array([82, 282, 482])  

    assert np.array_equal(peaks_pos, expected_peaks_pos)
