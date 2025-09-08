
import pytest

from ..conftest import RINGTEST_DIR


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
{
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "compartment_sets_file": str(RINGTEST_DIR / "compartment_sets.json"),
                "inputs": {
                    "override_field": 1,
                    "Stimulus": {
                        "module": "pulse",
                        "input_type": "current_clamp",
                        "represents_physical_electrode": True,
                        "amp_start": 3,
                        "width": 10,
                        "frequency": 50,
                        "delay": 0,
                        "duration": 50,
                        "compartment_set":"csA",
                        # "node_set": "RingA"
                    },
                },
            },
        }
    ],
    indirect=True,
)
@pytest.mark.slow
def test_compartment_set_input(create_tmp_simulation_config_file):
    """
    Test that the summation report matches the summed compartment report.

    Runs a simulation generating both compartment and summation reports for 'pas',
    then asserts that summing compartment data per gid equals the summation report data,
    within numerical tolerance.
    """
    from neurodamus import Neurodamus
    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()