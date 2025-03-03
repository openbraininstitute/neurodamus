import numpy as np
import pytest

def get_peak_idxs(trace, threshold=-0.5):
    """
    Identify peak indices in a signal by detecting strong negative second-order 
    derivatives after smoothing.

    Args:
        trace (array-like): Input signal.
        threshold (float, optional): Threshold for detecting strong reductions. 
                                     Defaults to -0.5.

    Returns:
        np.ndarray: Indices of detected peaks.
    """
    # Calculate the second-order difference of the 
    # voltage vector (trace_second_derivative)
    trace_second_derivative = np.diff(trace, 2)  
    # Convolve the trace_second_derivative with a smoothing kernel 
    # [1, 2, 4, 2, 1] to reduce noise
    window_sum = np.convolve(trace_second_derivative, [1, 2, 4, 2, 1], 'valid')  
    # Find the positions where the window sum is below threshold, 
    # indicating the beginning of a peak
    strong_reduction_pos = np.nonzero(window_sum < threshold)[0]  
    # Filter out consecutive positions, the negative second
    # derivative may persist for a while
    peaks_idxs = strong_reduction_pos[np.insert(
        np.diff(strong_reduction_pos) > 1, 0, True)]

    return peaks_idxs


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

    # extract the peak indexes
    peaks_pos = get_peak_idxs(voltage_vec)

    # Define expected positions where peaks are expected
    expected_peaks_pos = np.array([82, 282, 482])  

    assert np.array_equal(peaks_pos, expected_peaks_pos)




# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#             "node_set": "Mosaic",
#             "connection_overrides": [
#                 {
#                     "name": "A2A",
#                     "source": "RingA",
#                     "target": "RingA",
#                     "weight": 10001.1,
#                     "synapse_configure": "%s.Fac = 10002.1 %s.Dep = 10003.1",
#                     "delay": 0.0,
#                     "synapse_delay_override": 10005.1
#                 },
#                 {
#                     "name": "A2A_delayed",
#                     "source": "RingA",
#                     "target": "RingA",
#                     "weight": 10004.1,
#                     "synapse_configure": "%s.Fac = 10005.1",
#                     "delay": 1.0
#                 }
#             ]
#         },
#     },
# ], indirect=True)


# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "inputs": {
#                 "Stimulus": {
#                     "module": "pulse",
#                     "input_type": "current_clamp",
#                     "delay": 5,
#                     "duration": 50,
#                     "node_set": "RingA",
#                     "represents_physical_electrode": True,
#                     "amp_start": 10,
#                     "width": 1,
#                     "frequency": 50
#                 }
#             }
#         }
#     },
# ], indirect=True)
# @pytest.mark.forked
# def test_current_injection(create_tmp_simulation_config_file):
#     from neurodamus import Neurodamus
#     from neurodamus.core import NeurodamusCore as Nd

#     nd = Neurodamus(create_tmp_simulation_config_file)

#     cell_id = 1001
#     manager = nd.circuits.get_node_manager("RingB")
#     cell_ringB = manager.get_cell(cell_id)
#     voltage_vec = Nd.Vector()
#     voltage_vec.record(cell_ringB._cellref.soma[0](0.5)._ref_v)

#     Nd.finitialize()
#     nd.run()

#     # Calculate the second-order difference of the 
#     # voltage vector (v_increase_rate)
#     v_increase_rate = np.diff(voltage_vec, 2)  
#     # Convolve the v_increase_rate with a smoothing kernel 
#     # [1, 2, 4, 2, 1] to reduce noise
#     window_sum = np.convolve(v_increase_rate, [1, 2, 4, 2, 1], 'valid')  
#     # Find the positions where the window sum is below -0.5, 
#     # indicating the beginning of a peak in voltage_vec
#     strong_reduction_pos = np.nonzero(window_sum < -0.5)[0]  
#     # Filter out consecutive positions, the negative second
#     # derivative persists for a while
#     peaks_pos = strong_reduction_pos[np.insert(
#         np.diff(strong_reduction_pos) > 1, 0, True)]  
#     # Define expected positions where peaks are expected
#     expected_peaks_pos = np.array([82, 282, 482])  

#     assert np.array_equal(peaks_pos, expected_peaks_pos)


