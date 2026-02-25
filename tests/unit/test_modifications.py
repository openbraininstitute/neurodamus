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
        mod_manager.interpret(target_spec="dummy", mod_info=mod_info)


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
    n.enable_stimulus()

    soma_bef = n._pc.gid2cell(1).soma[0].gnabar_hh
    dend_bef = n._pc.gid2cell(1).dend[0].e_pas

    n.enable_modifications()

    soma_aft = n._pc.gid2cell(1).soma[0].gnabar_hh
    dend_aft = n._pc.gid2cell(1).dend[0].e_pas

    assert np.isclose(soma_bef, 0.12)
    assert np.isclose(dend_bef, -65.0)
    assert np.isclose(soma_aft, 11)
    assert np.isclose(dend_aft, 0.1)

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
                            "section_configure": "gnabar_hh = 3.0; cm = 2.0",
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
                    # "hh.gnabar" is equivalent to "gnabar_hh"
                    if hasattr(seg, "hh.gnabar"):
                        assert seg.hh.gnabar < 1.0

    n.enable_modifications()

    # Auxiliary structures to retrieve compartment set segments faster
    cs = n.target_manager.get_compartment_set("csA")
    comp_set = list(cs.filtered_iter(cs.node_ids()))

    # Matching segments for section-wise attributes (cm)
    cs_section_hits = set()
    # Matching segments for segment-specific attributes (hh.gnabar)
    cs_segment_hits = set()

    gid_to_manager = {}
    for cd in n.circuits.all_node_managers():
        for cell in cd.cells:
            gid_to_manager[cell.gid] = cd

    for cl in comp_set:
        cs_section_hits.add((cl.node_id, cl.section_id))

        section = gid_to_manager[cl.node_id].get_cell(cl.node_id).get_sec(cl.section_id)
        segment = section(cl.offset)
        # Segment node index should be unique for each segment in the circuit
        # The same segment can be obtained when indexing through different offsets,
        # depending on the number of segments (nsec)
        seg_nidx = segment.node_index()
        cs_segment_hits.add((cl.node_id, cl.section_id, seg_nidx))

    for cd in n.circuits.all_node_managers():
        for cell in cd.cells:
            for sec_id in set(cell.get_section_counts()):
                sec = cell.get_sec(sec_id)
                for seg in sec.allseg():
                    in_section = (cell.gid, sec_id) in cs_section_hits
                    in_segment = (cell.gid, sec_id, seg.node_index()) in cs_segment_hits

                    if hasattr(seg, "cm"):
                        assert np.isclose(seg.cm, 2.) if in_section else np.isclose(seg.cm, 1.)

                    if hasattr(seg, "hh.gnabar"):
                        assert np.isclose(seg.hh.gnabar, 3.) if in_segment else seg.hh.gnabar < 1.
