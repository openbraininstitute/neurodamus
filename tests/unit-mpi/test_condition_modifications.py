import json
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from mpi4py import MPI

from tests.conftest import RINGTEST_DIR

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.parametrize(
    "simulator",
    ["NEURON", "CORENEURON"],
)
@pytest.mark.parametrize(
    ("mod_type", "target_param", "target_name", "section_configure", "expected"),
    [
        (
            "section_list",
            "node_set",
            "RingA:oneCell",
            "somatic.gnabar_hh *= 2.; basal.gnabar_hh += 1.",
            {"soma": {"gnabar_hh": [0.24]}, "dend": {"gnabar_hh": [1.12, 1.12]}},
        ),
        (
            "section",
            "node_set",
            "RingA:oneCell",
            "soma[0].gnabar_hh *= 2.; dend[0].gnabar_hh += 1.",
            {"soma": {"gnabar_hh": [0.24]}, "dend": {"gnabar_hh": [1.12, 0.12]}},
        ),
        (
            "compartment_set",
            "compartment_set",
            "csA",
            "gnabar_hh *= 2.; gnabar_hh += 1.",
            {"soma": {"gnabar_hh": [1.24]}, "dend": {"gnabar_hh": [0.12, 1.24]}},
        ),
    ],
)
@pytest.mark.mpi(ranks=2)
def test_modifications_with_neuron_and_coreneuron_mpi_multiple_types(
    ringtest_baseconfig,
    simulator,
    mod_type,
    target_param,
    target_name,
    section_configure,
    expected,
    mpi_ranks,
):
    """
    Test condition modifications (section_list, section, and compartment_set) running NEURON and
    CoreNeuron with MPI.
    """
    assert size == mpi_ranks

    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Nd

    modif_config = ringtest_baseconfig
    modif_config["target_simulator"] = simulator

    # Build modification dict dynamically
    modification = {
        "name": "augassign_test",
        "type": mod_type,
        target_param: target_name,
        "section_configure": section_configure,
    }

    modif_config["conditions"]["modifications"] = [modification]

    # Add compartment sets file if compartment_set modification
    if target_param == "compartment_set":
        cs_path = str(RINGTEST_DIR) + "/compartment_sets.json"
        modif_config["compartment_sets_file"] = cs_path

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    n = Neurodamus(sim_file_path)

    # Check if the MPI rank has the cell
    if n._pc.gid_exists(0):
        # Cell 0 is not in the node_set/compartment_set, so it is not modified
        cell0 = Nd._pc.gid2cell(0)
        assert np.isclose(cell0.soma[0].gnabar_hh, 0.12)
        assert np.isclose(cell0.dend[0].gnabar_hh, 0.12)
        assert np.isclose(cell0.dend[1].gnabar_hh, 0.12)

    if n._pc.gid_exists(1):
        # Cell 1 is modified
        cell1 = Nd._pc.gid2cell(1)
        for sec_type in expected:
            # soma, dend
            val_dict = expected[sec_type]
            for mech in val_dict:
                # gnabar_hh
                values = val_dict[mech]
                for idx, val in enumerate(values):
                    section = getattr(cell1, sec_type)[idx]
                    mechanism = getattr(section, mech)
                    assert np.isclose(mechanism, val)

    n.run()
