
import pytest

from ..conftest import RINGTEST_DIR

from collections import Counter


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
                        "compartment_set": "csA",
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
    Verify that the stimulus defined in the config is applied only to the
    specified compartment set (csA). For each cell section, the number of
    IClamp point processes created in NEURON must match the expected count
    from the compartment set definition.
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
                assert count == clamps.get((cell.gid, section_id), 0)


