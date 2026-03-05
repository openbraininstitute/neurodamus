import json
import re
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from neurodamus.core.configuration import ConfigurationError
from neurodamus.modification_manager import ModificationManager
from neurodamus.node import Neurodamus, Node
from types import SimpleNamespace

from tests.conftest import RINGTEST_DIR

SIMULATION_CONFIG_FILE = RINGTEST_DIR / "simulation_config.json"

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "conditions": {
                    "modifications": [
                        {
                            "name": "applyTTX",
                            "type": "ttx",
                            "node_set": "RingA"
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_applyTTX(create_tmp_simulation_config_file):
    """
    A test of enabling TTX with a short simulation.
    As Ringtest cells don't contain mechanisms that use the TTX concentration
    to enable/disable sodium channels, no spike change is expected.
    Instead, we check that all sections contain TTXDynamicsSwitch after modification
    """

    # NeuronWrapper needs to be imported at function level
    from neurodamus.core import NeuronWrapper as Nd

    n = Node(create_tmp_simulation_config_file)

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()

    # check TTXDynamicsSwitch is not inserted in any section of cell gid=1
    cell = n._pc.gid2cell(1)
    for sec in cell.all:
        assert not Nd.ismembrane("TTXDynamicsSwitch", sec=sec)

    n.enable_modifications()

    # check TTXDynamicsSwitch is inserted after modifications
    for sec in cell.all:
        assert Nd.ismembrane("TTXDynamicsSwitch", sec=sec)

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "no_SK_E2",
                            "node_set": "Mosaic",
                            "type": "configure_all_sections",
                            "section_configure": "%s.gnabar_hh = 0",
                        }
                    ]
                },
                "inputs": {
                    "pulse": {
                        "module": "pulse",
                        "input_type": "current_clamp",
                        "delay": 5,
                        "duration": 50,
                        "node_set": "RingA",
                        "amp_start": 5,
                        "width": 1,
                        "frequency": 50,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_configure_all_sections(create_tmp_simulation_config_file):
    """
    A test of performing configure_all_sections with a short simulation.
    Without the modification, there are spikes with the given stimulus.
    After applying sec.gnabar_hh = 0, the expected outcome is 0 spike.
    """

    # NeuronWrapper needs to be imported at function level
    from neurodamus.core import NeuronWrapper as Nd

    n = Node(create_tmp_simulation_config_file)
    sec_variable = "gnabar_hh"

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    n.enable_stimulus()
    n.sim_init()
    n.solve()
    nspike_no_configure_all_sections = sum(len(spikes) for spikes, _ in n._spike_vecs)

    # check section variable initial value before modification
    cell = n._pc.gid2cell(1)
    for sec in cell.all:
        assert getattr(sec, sec_variable) > 0

    n.enable_modifications()

    # check section variable value after modifications
    for sec in cell.all:
        assert getattr(sec, sec_variable) == 0

    # setup sim again
    Nd.t = 0.0
    n._sim_ready = False
    n.sim_init()
    n.solve()
    nspike_configure_all_sections = sum(len(spikes) for spikes, _ in n._spike_vecs)

    assert nspike_no_configure_all_sections > 0
    assert nspike_configure_all_sections == 0


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "no_SK_E2",
                            "node_set": "RingA:oneCell",
                            "type": "configure_all_sections",
                            "section_configure": "%s.gnabar_hh *= 11; %s.e_pas *= 0.1",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_configure_all_sections_AugAssign(create_tmp_simulation_config_file):
    """Test the augmented assignment (*=) and multiple assignments for configure_all_sections"""

    # NeuronWrapper needs to be imported at function level
    from neurodamus.core import NeuronWrapper as Nd

    Neurodamus(create_tmp_simulation_config_file)
    soma1 = Nd._pc.gid2cell(0).soma[0]
    soma2 = Nd._pc.gid2cell(1).soma[0]

    assert np.isclose(soma1.gnabar_hh, 0.12)
    assert np.isclose(soma1.e_pas, -70)
    assert np.isclose(soma2.gnabar_hh, 1.32)
    assert np.isclose(soma2.e_pas, -7)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "no_SK_E2",
                            "node_set": "RingA:oneCell",
                            "type": "configure_all_sections",
                            "section_configure": "%s.e_pas *= 0.1",
                        },
                        {
                            "name": "no_SK_E2",
                            "node_set": "RingA:oneCell",
                            "type": "configure_all_sections",
                            "section_configure": "%s.gnabar_hh *= 11",
                        }

                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_configure_all_sections_AugAssign_name_clash(create_tmp_simulation_config_file):
    """This should produce the same results as test_configure_all_sections_AugAssign
    
    However, here we apply the same modification in 2 steps with modifications 
    that have the same name. Their combined effect should be equivalent 
    to the modification in test_configure_all_sections_AugAssign.
    """

    # NeuronWrapper needs to be imported at function level
    from neurodamus.core import NeuronWrapper as Nd

    Neurodamus(create_tmp_simulation_config_file)
    soma1 = Nd._pc.gid2cell(0).soma[0]
    soma2 = Nd._pc.gid2cell(1).soma[0]

    assert np.isclose(soma1.gnabar_hh, 0.12)
    assert np.isclose(soma1.e_pas, -70)
    assert np.isclose(soma2.gnabar_hh, 1.32)
    assert np.isclose(soma2.e_pas, -7)

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "no_SK_E2",
                            "node_set": "Mosaic",
                            "type": "configure_all_sections",
                            "section_configure": "%s.gSK_E2bar_SK_E2 = 0",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_warning_no_modification(create_tmp_simulation_config_file, capsys):
    """Test warning when modification is not applied on any section"""
    Neurodamus(create_tmp_simulation_config_file)
    captured = capsys.readouterr()
    assert "configure_all_sections applied to zero sections" in captured.out


def test_error_unknown_modification():
    """Test error handling: unknown modification type"""
    mod_manager = ModificationManager(target_manager="dummy")
    with pytest.raises(ConfigurationError, match="Unknown Modification mod_blabla"):
        mod_manager.interpret(target_spec="dummy", mod_info={"Type": "mod_blabla"})

def test_error_unknown_modification():
    mod_manager = ModificationManager(target_manager="dummy")
    unknown_type = object()
    mod_info = SimpleNamespace(type=unknown_type)

    with pytest.raises(
        ConfigurationError,
        match="Unknown Modification",
    ):
        mod_manager.interpret(mod_info=mod_info)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "no_SK_E2",
                            "node_set": "Mosaic",
                            "type": "configure_all_sections",
                            "section_configure": "gSK_E2bar_SK_E2 = 0",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_error_wrong_section_configure_syntax(create_tmp_simulation_config_file):
    """Test error handling: wrong section_configure syntax"""
    with pytest.raises(
        ConfigurationError,
        match="section_configure only supports single assignments of "
        "attributes of the section wildcard %s",
    ):
        Neurodamus(create_tmp_simulation_config_file)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "no_SK_E2",
                            "node_set": "Mosaic",
                            "type": "configure_all_sections",
                            "section_configure": "print(gSK_E2bar_SK_E2)",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_error_wrong_invalid_operation(create_tmp_simulation_config_file):
    """Test error handling: wrong operation (neither assign nor aug assign)"""
    with pytest.raises(
        ConfigurationError,
        match="section_configure must consist of one or more semicolon-separated assignments",
    ):
        Neurodamus(create_tmp_simulation_config_file)

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "scale_soma",
                            "node_set": "Mosaic",
                            "type": "section_list",
                            "section_configure": "somatic.gnabar_hh = 0; basal.gnabar_hh = 0",
                        }
                    ]
                },
                "inputs": {
                    "pulse": {
                        "module": "pulse",
                        "input_type": "current_clamp",
                        "delay": 5,
                        "duration": 50,
                        "node_set": "RingA",
                        "amp_start": 5,
                        "width": 1,
                        "frequency": 50,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_modification_section_list(create_tmp_simulation_config_file):
    """
    A test of performing section_list with a short simulation.
    Without the modification, there are spikes with the given stimulus.
    After applying apical.gnabar_hh = 0, the expected outcome is 0 spikes.
    """

    # NeuronWrapper needs to be imported at function level
    from neurodamus.core import NeuronWrapper as Nd

    n = Node(create_tmp_simulation_config_file)
    sec_variable = "gnabar_hh"

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    n.enable_stimulus()
    n.sim_init()
    n.solve()
    nspike_no_modif = sum(len(spikes) for spikes, _ in n._spike_vecs)

    # check section variable initial value before modification
    cell = n._pc.gid2cell(1)
    for sec in cell.all:
        assert getattr(sec, sec_variable) > 0

    n.enable_modifications()

    # check section variable value after modifications
    for sec in cell.all:
        assert getattr(sec, sec_variable) == 0

    # setup sim again
    Nd.t = 0.0
    n._sim_ready = False
    n.sim_init()
    n.solve()
    nspike_modif_section_list = sum(len(spikes) for spikes, _ in n._spike_vecs)

    assert nspike_no_modif > 0
    assert nspike_modif_section_list == 0


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "no_SK_E2",
                            "node_set": "RingA:oneCell",
                            "type": "section",
                            "section_configure": "soma[0].gnabar_hh = 11; dend[0].e_pas = 0.1",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_modification_section(create_tmp_simulation_config_file):
    """Test the augmented assignment (*=) and multiple assignments for configure_all_sections"""

    n = Node(create_tmp_simulation_config_file)

    # setup sim config
    n.load_targets()
    n.create_cells()
    n.create_synapses()

    soma = n._pc.gid2cell(1).soma[0].gnabar_hh
    dend = n._pc.gid2cell(1).dend[0].e_pas

    n.enable_modifications()

    soma_mod = n._pc.gid2cell(1).soma[0].gnabar_hh
    dend_mod = n._pc.gid2cell(1).dend[0].e_pas

    assert np.isclose(soma, 0.12)
    assert np.isclose(dend, -65.0)
    assert np.isclose(soma_mod, 11)
    assert np.isclose(dend_mod, 0.1)

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "compartment_sets_file": str(RINGTEST_DIR / "compartment_sets.json"),
                "conditions": {
                    "modifications": [
                        {
                            "name": "Ca_hotspot_dend[10]_manipulation",
                            "compartment_set": "csA",
                            "type": "compartment_set",
                            "section_configure": "gnabar_hh = -3.0; cm = 2.0",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_modification_compartment_set(create_tmp_simulation_config_file):
    """Test the augmented assignment (*=) and multiple assignments for compartment_set"""

    from neurodamus.core import NeuronWrapper as Nd
    n = Node(create_tmp_simulation_config_file)

    # setup sim config
    n.load_targets()
    n.create_cells()
    n.create_synapses()

    # Check values before applying modifications
    for cd in n.circuits.all_node_managers():
        for cell in cd.cells:
            cell_sections = set(cell.get_section_counts())
            for section_id in cell_sections:
                sec = cell.get_sec(section_id)
                for seg in sec.allseg():
                    if hasattr(seg, "cm"):
                        assert np.isclose(seg.cm, 1.0)
                    if hasattr(seg, "gnabar_hh"):
                        assert 0.0 < seg.gnabar_hh < 1.0

    n.enable_modifications()

    # Auxiliary structures to retrieve compartment set segments faster
    cs = n.target_manager.get_compartment_set("csA")
    comp_set = list(cs.filtered_iter(cs.node_ids()))

    # Set of segments that will be modified
    modif_segments = set()

    gid_to_manager = {}
    for cd in n.circuits.all_node_managers():
        for cell in cd.cells:
            gid_to_manager[cell.gid] = cd

    for cl in comp_set:
        section = gid_to_manager[cl.node_id].get_cell(cl.node_id).get_sec(cl.section_id)
        segment = section(cl.offset)
        # Segment node index should be unique for each segment in the circuit
        # The same segment can be obtained when indexing through different offsets,
        # depending on the number of segments (nsec)
        seg_nidx = segment.node_index()
        modif_segments.add((cl.node_id, cl.section_id, seg_nidx))

    # We do a second pass on the sections, as compartments at the beginning/end may be modified:
    # - For offset=0, it will take the value of the next compartment
    # - For offset=1, it will take the value of the previous compartment
    orig_modif_segments = modif_segments.copy()
    for node_id, sec_id, node_idx in orig_modif_segments:
        section = gid_to_manager[node_id].get_cell(node_id).get_sec(sec_id)
        # We store the previous compartment for the next iteration, in case we:
        # - need to add it
        # - check it was modified
        prev_comp = None
        for seg in section.allseg():
            if (node_id, sec_id, seg.node_index()) in orig_modif_segments:
                if prev_comp is not None and np.isclose(prev_comp.x, 0.):
                    # If current compartment is modified and previous compartment was 0
                    modif_segments.add((node_id, sec_id, prev_comp.node_index()))

            if np.isclose(seg.x, 1.):
                # This is the last compartment, check if the previous one was modified
                if prev_comp is not None and (
                        (node_id, sec_id, prev_comp.node_index()) in orig_modif_segments
                ):
                    modif_segments.add((node_id, sec_id, seg.node_index()))
            prev_comp = seg

    # Check again all values after applying modifications
    for cd in n.circuits.all_node_managers():
        for cell in cd.cells:
            for sec in cell.CellRef.all:
                sec_id = cell.get_section_id(sec)
                for seg in sec.allseg():
                    is_modif = (cell.gid, sec_id, seg.node_index()) in modif_segments

                    if hasattr(seg, "cm"):
                        assert np.isclose(seg.cm, 2.) if is_modif else np.isclose(seg.cm, 1.)

                    if hasattr(seg, "gnabar_hh"):
                        assert np.isclose(seg.gnabar_hh, -3) if is_modif else 0 < seg.gnabar_hh < 1

@pytest.mark.parametrize(
    "mod_type, target_param, target_name, section_configure, expected",
    [
        (
            "section_list",
            "node_set",
            "RingA:oneCell",
            "somatic.gnabar_hh *= 2.; basal.gnabar_hh += 1.; somatic.Ra -= 50.; basal.Ra /= 10.",
            {
                "soma": {"gnabar_hh": [0.24], "Ra": [50.]},
                "dend": {"gnabar_hh": [1.12, 1.12], "Ra": [10., 10.]}
            }
         ),
        (
            "section",
            "node_set",
            "RingA:oneCell",
            "soma[0].gnabar_hh *= 2.; dend[0].gnabar_hh += 1.; soma[0].Ra -= 50.; dend[1].Ra /= 10.",
            {
                "soma": {"gnabar_hh": [0.24], "Ra": [50.]},
                "dend": {"gnabar_hh": [1.12, 0.12], "Ra": [100., 10.]}
            }
        ),
        (
            "compartment_set",
            "compartment_set",
            "csA",
            "gnabar_hh *= 2.; gnabar_hh += 1.; e_pas -= 5.; e_pas /= 10.",
            {
                "soma": {"gnabar_hh": [1.24], "e_pas": [-7.5]},
                "dend": {"gnabar_hh": [0.12, 1.24], "e_pas": [-65., -7.]}
            }
        ),
    ],
)
def test_modification_augassign_multiple_types(
        ringtest_baseconfig, mod_type, target_param, target_name, section_configure, expected
):
    """
    Test augmented assignments (+=, -=, *=, /=) for section_list, section, and compartment_set.
    """
    # NeuronWrapper needs to be imported at function level
    from neurodamus.core import NeuronWrapper as Nd

    # Build modification dict dynamically
    modification = {
        "name": "augassign_test",
        "type": mod_type,
        target_param: target_name,
        "section_configure": section_configure,
    }

    modif_config = ringtest_baseconfig
    modif_config["conditions"]["modifications"] = [modification]

    # Add compartment sets file if compartment_set modification
    if target_param == "compartment_set":
        cs_path = str(RINGTEST_DIR) + "/compartment_sets.json"
        modif_config["compartment_sets_file"] = cs_path

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    # Process simulation config
    n = Node(sim_file_path)

    # setup sim config
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    n.enable_modifications()

    # Retrieve cells
    cell0 = Nd._pc.gid2cell(0)
    cell1 = Nd._pc.gid2cell(1)

    # Cell 0 is not in the node_set/compartment_set, so it is not modified
    assert np.isclose(cell0.soma[0].gnabar_hh, 0.12)
    assert np.isclose(cell0.dend[0].gnabar_hh, 0.12)
    assert np.isclose(cell0.dend[1].gnabar_hh, 0.12)

    # Cell 1 is modified
    for type in expected.keys():
        # soma, dend
        val_dict = expected[type]
        for mech in val_dict.keys():
            # gnabar_hh, Ra/e_pas
            values = val_dict[mech]
            for idx, val in enumerate(values):
                section = getattr(cell1, type)[idx]
                mechanism = getattr(section, mech)
                assert np.isclose(mechanism, val)

@pytest.mark.parametrize(
    "mod_type, section_configure",
    [
        ("section_list", "somatic.gSK_E2bar_SK_E2 = 0"),
        ("section", "soma[0].gSK_E2bar_SK_E2 = 0"),
        ("compartment_set", "gSK_E2bar_SK_E2 = 0"),
    ],
)
def test_modification_no_modif_multiple_types(
        ringtest_baseconfig, capsys, mod_type, section_configure
):
    """Test warning when modification is not applied on any target."""

    # Build modification dict dynamically
    # We can add both node_set and compartment_set, the irrelevant one is just ignored
    modification = {
        "name": "no_modif_test",
        "type": mod_type,
        "node_set": "Mosaic",
        "compartment_set": "csA",
        "section_configure": section_configure,
    }

    modif_config = ringtest_baseconfig
    modif_config["conditions"]["modifications"] = [modification]

    # Add compartment sets file if compartment_set modification
    if mod_type == "compartment_set":
        cs_path = str(RINGTEST_DIR) + "/compartment_sets.json"
        modif_config["compartment_sets_file"] = cs_path

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    # Process simulation config
    Neurodamus(sim_file_path)
    captured = capsys.readouterr()
    if mod_type == "compartment_set":
        assert f"{mod_type} applied to zero segments" in captured.out
    else:
        assert f"{mod_type} applied to zero sections" in captured.out

@pytest.mark.parametrize(
    "mod_type, target, section_configure",
    [
        ("section_list", "compartment_set", "somatic.gSK_E2bar_SK_E2 = 0"),
        ("section", "compartment_set", "soma[0].gSK_E2bar_SK_E2 = 0"),
        ("compartment_set", "node_set", "gSK_E2bar_SK_E2 = 0"),
    ],
)
def test_modification_wrong_target_multiple_types(
        ringtest_baseconfig, mod_type, target, section_configure
):
    """Test error handling: missing appropriate node_set/compartment_set keys."""

    # Build modification dict dynamically
    modification = {
        "name": "wrong_target_test",
        "type": mod_type,
        target: "Mosaic",
        "section_configure": section_configure,
    }

    modif_config = ringtest_baseconfig
    modif_config["conditions"]["modifications"] = [modification]

    # Add compartment sets file if compartment_set modification
    if mod_type == "compartment_set":
        cs_path = str(RINGTEST_DIR) + "/compartment_sets.json"
        modif_config["compartment_sets_file"] = cs_path

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    if mod_type == "compartment_set":
        error_message = "Could not find 'compartment_set' in 'modification 0'"
    else:
        error_message = "Could not find 'node_set' in 'modification 0'"

    with pytest.raises(
        Exception,
        match=error_message,
    ):
        Neurodamus(sim_file_path)

@pytest.mark.parametrize(
    "mod_type, section_configure",
    [
        ("section_list", "print(somatic.gSK_E2bar_SK_E2)"),
        ("section", "print(soma[0].gSK_E2bar_SK_E2)"),
        ("compartment_set", "print(gSK_E2bar_SK_E2)"),
    ],
)
def test_modification_wrong_syntax_no_assignments_multiple_types(
        ringtest_baseconfig, mod_type, section_configure
):
    """Test wrong syntax in section_configure: no assignments."""

    # Build modification dict dynamically
    # We can add both node_set and compartment_set, the irrelevant one is just ignored
    modification = {
        "name": "wrong_syntax_test",
        "type": mod_type,
        "node_set": "Mosaic",
        "compartment_set": "csA",
        "section_configure": section_configure,
    }

    modif_config = ringtest_baseconfig
    modif_config["conditions"]["modifications"] = [modification]

    # Add compartment sets file if compartment_set modification
    if mod_type == "compartment_set":
        cs_path = str(RINGTEST_DIR) + "/compartment_sets.json"
        modif_config["compartment_sets_file"] = cs_path

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    with pytest.raises(
        ConfigurationError,
        match="section_configure must contain assignments only"
    ):
        # Process simulation config
        Neurodamus(sim_file_path)

@pytest.mark.parametrize(
    "mod_type, section_configure",
    [
        ("section_list", "somatic.gSK_E2bar_SK_E2 = somatic.gSK_E2bar_SK_E2 = 0"),
        ("section", "soma[0].gSK_E2bar_SK_E2 = soma[0].gSK_E2bar_SK_E2 = 0"),
        ("compartment_set", "gSK_E2bar_SK_E2 = gSK_E2bar_SK_E2 = 0"),
    ],
)
def test_modification_wrong_syntax_multiple_assignments_multiple_types(
        ringtest_baseconfig, mod_type, section_configure
):
    """Test wrong syntax in section_configure: chained assignments."""

    # Build modification dict dynamically
    # We can add both node_set and compartment_set, the irrelevant one is just ignored
    modification = {
        "name": "wrong_syntax_test",
        "type": mod_type,
        "node_set": "Mosaic",
        "compartment_set": "csA",
        "section_configure": section_configure,
    }

    modif_config = ringtest_baseconfig
    modif_config["conditions"]["modifications"] = [modification]

    # Add compartment sets file if compartment_set modification
    if mod_type == "compartment_set":
        cs_path = str(RINGTEST_DIR) + "/compartment_sets.json"
        modif_config["compartment_sets_file"] = cs_path

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    with pytest.raises(
        ConfigurationError,
        match="Only single-target assignments are supported in section_configure"
    ):
        # Process simulation config
        Neurodamus(sim_file_path)

@pytest.mark.parametrize(
    "mod_type, section_configure",
    [
        ("section_list", "somatic.gSK_E2bar_SK_E2 = somatic.gSK_E2bar_SK_E2"),
        ("section", "soma[0].gSK_E2bar_SK_E2 = soma[0].gSK_E2bar_SK_E2"),
        ("compartment_set", "gSK_E2bar_SK_E2 = gSK_E2bar_SK_E2"),
    ],
)
def test_modification_wrong_syntax_non_num_const_multiple_types(
        ringtest_baseconfig, mod_type, section_configure
):
    """Test wrong syntax in section_configure: non-numeric constants."""

    # Build modification dict dynamically
    # We can add both node_set and compartment_set, the irrelevant one is just ignored
    modification = {
        "name": "wrong_syntax_test",
        "type": mod_type,
        "node_set": "Mosaic",
        "compartment_set": "csA",
        "section_configure": section_configure,
    }

    modif_config = ringtest_baseconfig
    modif_config["conditions"]["modifications"] = [modification]

    # Add compartment sets file if compartment_set modification
    if mod_type == "compartment_set":
        cs_path = str(RINGTEST_DIR) + "/compartment_sets.json"
        modif_config["compartment_sets_file"] = cs_path

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    with pytest.raises(
        ConfigurationError,
        match="Only numeric constants are allowed in section_configure"
    ):
        # Process simulation config
        Neurodamus(sim_file_path)

@pytest.mark.parametrize(
    "mod_type, section_configure",
    [
        ("section_list", "somatic = 0"),
        ("section", "soma[0] = 0"),
        ("compartment_set", "gnabar[0] = 0"),
    ],
)
def test_modification_wrong_syntax_no_attr_multiple_types(
        ringtest_baseconfig, mod_type, section_configure
):
    """Test wrong syntax in section_configure: no attribute in assignment."""

    # Build modification dict dynamically
    # We can add both node_set and compartment_set, the irrelevant one is just ignored
    modification = {
        "name": "wrong_syntax_test",
        "type": mod_type,
        "node_set": "Mosaic",
        "compartment_set": "csA",
        "section_configure": section_configure,
    }

    modif_config = ringtest_baseconfig
    modif_config["conditions"]["modifications"] = [modification]

    # Add compartment sets file if compartment_set modification
    if mod_type == "compartment_set":
        cs_path = str(RINGTEST_DIR) + "/compartment_sets.json"
        modif_config["compartment_sets_file"] = cs_path

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    if mod_type == "compartment_set":
        err_msg = "target variables like "
    else:
        err_msg = "use syntax like "

    with pytest.raises(
        ConfigurationError,
        match=f"{mod_type} modification must {err_msg}",
    ):
        # Process simulation config
        Neurodamus(sim_file_path)

@pytest.mark.parametrize(
    "mod_type, section_configure",
    [
        ("section_list", "soma.gnabar_hh = 0"),
        ("section", "somatic[0].gnabar_hh = 0"),
    ],
)
def test_modification_invalid_section_multiple_types(
        ringtest_baseconfig, mod_type, section_configure
):
    """Test passing invalid section name (e.g.: different from somatic/soma, basal/dend, etc.)."""

    # Build modification dict dynamically
    # We can add both node_set and compartment_set, the irrelevant one is just ignored
    modification = {
        "name": "invalid_section_test",
        "type": mod_type,
        "node_set": "Mosaic",
        "section_configure": section_configure,
    }

    modif_config = ringtest_baseconfig
    modif_config["conditions"]["modifications"] = [modification]

    # Write to temp JSON file
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(modif_config, f, indent=2)
        sim_file_path = f.name

    if mod_type == "section_list":
        err_msg = \
            "Unknown section type: soma. Allowed types are: all, apical, axonal, basal, somatic"
    else:
        err_msg = "Unknown section type: somatic. Allowed types are: apic, axon, dend, soma"

    with pytest.raises(
        ConfigurationError,
        match=f"{err_msg}",
    ):
        # Process simulation config
        Neurodamus(sim_file_path)

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "wrong_syntax_test",
                            "node_set": "Mosaic",
                            "type": "section",
                            "section_configure": "soma[3].gnabar_hh = 0",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_modification_idx_oob_section(create_tmp_simulation_config_file, capsys):
    """Test section index out of bounds: shouldn't crash, just warn about 0 modifications applied."""

    Neurodamus(create_tmp_simulation_config_file)
    captured = capsys.readouterr()
    assert "section applied to zero sections" in captured.out

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "conditions": {
                    "modifications": [
                        {
                            "name": "wrong_syntax_test",
                            "node_set": "Mosaic",
                            "type": "section",
                            "section_configure": "soma.gnabar_hh = 0",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_modification_no_idx_section(create_tmp_simulation_config_file):
    """Test wrong syntax: non-indexed section."""
    with pytest.raises(
        ConfigurationError,
        match=re.escape("Section must be indexed"),
    ):
        Neurodamus(create_tmp_simulation_config_file)
