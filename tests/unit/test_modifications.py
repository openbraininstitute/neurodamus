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
