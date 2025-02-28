import pytest
import numpy


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "inputs": {
                "Stimulus": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 1,
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

    v_increase_rate = numpy.diff(voltage_vec, 2)
    window_sum = numpy.convolve(v_increase_rate, [1, 2, 4, 2, 1], 'valid')
    strong_reduction_pos = numpy.nonzero(window_sum < -0.5)[0]
    non_consecutives_pos = strong_reduction_pos[numpy.insert(
        numpy.diff(strong_reduction_pos) > 1, 0, True)]
    expected_positions = numpy.array([42, 242, 442])

    assert numpy.array_equal(non_consecutives_pos, expected_positions)
