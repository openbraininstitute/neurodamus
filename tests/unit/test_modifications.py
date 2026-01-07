import numpy as np
import pytest

from tests.conftest import RINGTEST_DIR

from neurodamus.core.configuration import ConfigurationError, SimConfig
from neurodamus.modification_manager import ModificationManager
from neurodamus.node import Neurodamus, Node

SIMULATION_CONFIG_FILE = RINGTEST_DIR / "simulation_config.json"


def test_applyTTX():
    """
    A test of enabling TTX with a short simulation.
    As Ringtest cells don't contain mechanisms that use the TTX concentration
    to enable/disable sodium channels, no spike change is expected.
    Instead, we check that all sections contain TTXDynamicsSwitch after modification
    """

    # NeuronWrapper needs to be imported at function level
    from neurodamus.core import NeuronWrapper as Nd

    n = Node(str(SIMULATION_CONFIG_FILE))

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()

    # check TTXDynamicsSwitch is not inserted in any section of cell gid=1
    cell = n._pc.gid2cell(1)
    for sec in cell.all:
        assert not Nd.ismembrane("TTXDynamicsSwitch", sec=sec)

    # append modification to config directly
    TTX_mod = {"Name": "applyTTX", "Type": "TTX", "Target": "RingA"}
    SimConfig.modifications.append(TTX_mod)

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
def test_ConfigureAllSections(create_tmp_simulation_config_file):
    """
    A test of performing ConfigureAllSections with a short simulation.
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
    nspike_noConfigureAllSections = sum(len(spikes) for spikes, _ in n._spike_vecs)

    # check section variable initial value before modification
    cell = n._pc.gid2cell(1)
    for sec in cell.all:
        assert getattr(sec, sec_variable) > 0

    # append modification to config directly
    ConfigureAllSections_mod = {
        "Name": "no_SK_E2", 
        "Type": "ConfigureAllSections",
        "Target": "Mosaic",
        "SectionConfigure": f"%s.{sec_variable} = 0",
    }
    SimConfig.modifications.append(ConfigureAllSections_mod)

    n.enable_modifications()

    # check section variable value after modifications
    for sec in cell.all:
        assert getattr(sec, sec_variable) == 0

    # setup sim again
    Nd.t = 0.0
    n._sim_ready = False
    n.sim_init()
    n.solve()
    nspike_ConfigureAllSections = sum(len(spikes) for spikes, _ in n._spike_vecs)

    assert nspike_noConfigureAllSections > 0
    assert nspike_ConfigureAllSections == 0


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
                            "type": "ConfigureAllSections",
                            "section_configure": "%s.gnabar_hh *= 11; %s.e_pas *= 0.1",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_ConfigureAllSections_AugAssign(create_tmp_simulation_config_file):
    """Test the augmented assignment (*=) and multiple assignments for ConfigureAllSections"""

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
                            "type": "ConfigureAllSections",
                            "section_configure": "%s.e_pas *= 0.1",
                        },
                        {
                            "name": "no_SK_E2",
                            "node_set": "RingA:oneCell",
                            "type": "ConfigureAllSections",
                            "section_configure": "%s.gnabar_hh *= 11",
                        }

                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_ConfigureAllSections_AugAssign_name_clash(create_tmp_simulation_config_file):
    """This should produce the same results as test_ConfigureAllSections_AugAssign
    
    However, here we apply the same modification in 2 steps with modifications 
    that have the same name. Their combined effect should be equivalent 
    to the modification in test_ConfigureAllSections_AugAssign.
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
                            "type": "ConfigureAllSections",
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
    assert "ConfigureAllSections applied to zero sections" in captured.out


def test_error_unknown_modification():
    """Test error handling: unknown modification type"""
    mod_manager = ModificationManager(target_manager="dummy")
    with pytest.raises(ConfigurationError, match="Unknown Modification mod_blabla"):
        mod_manager.interpret(target_spec="dummy", mod_info={"Type": "mod_blabla"})


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
                            "type": "ConfigureAllSections",
                            "section_configure": "gSK_E2bar_SK_E2 = 0",
                        }
                    ]
                },
            },
        }
    ],
    indirect=True,
)
def test_error_wrong_SectionConfigure_syntax(create_tmp_simulation_config_file):
    """Test error handling: wrong SectionConfigure syntax"""
    with pytest.raises(
        ConfigurationError,
        match="SectionConfigure only supports single assignments of "
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
                            "type": "ConfigureAllSections",
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
        match="SectionConfigure must consist of one or more semicolon-separated assignments",
    ):
        Neurodamus(create_tmp_simulation_config_file)
