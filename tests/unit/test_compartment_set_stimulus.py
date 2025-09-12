
import pytest

from ..conftest import RINGTEST_DIR

from collections import Counter


def inspect(v):
    print(v, type(v))
    for i in dir(v):
        print(i)

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
    cs = nd.target_manager.get_compartment_set("csA")
    clamps = Counter((cl.node_id, cl.section_id) for cl in cs)

    for cd in nd.circuits.all_node_managers():
        for cell in cd.cells:
            for section_id in range(3):
                sec = cell.get_sec(section_id)
                count = sum(
                    1
                    for seg in sec.allseg()
                    for pp in seg.point_processes()
                    if "IClamp" in pp.hname() and seg.x > 0 and seg.x < 1
                )
                assert count == clamps.get((cell.gid-1, section_id), 0)


