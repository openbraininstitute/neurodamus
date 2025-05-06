"""Test all configurable parameters of `connection_overrides` in `simulation_config.json`.

Reference:
https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html#connection-overrides
"""


import numpy as np
import pytest

from tests import utils


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "connection_overrides": [
                {
                    "name": "A2A",
                    "source": "RingA",
                    "target": "RingA",
                    "weight": 1001.1,
                    "synapse_configure": "%s.Fac = 1002.1 %s.Dep = 1003.1",
                    "delay": 0.0,
                    "synapse_delay_override": 1004.1
                },
                {
                    "name": "A2A_delayed",
                    "source": "RingA",
                    "target": "RingA",
                    "weight": 1005.1,
                    "synapse_configure": "%s.Fac = 1005.1",
                    "delay": 1.0
                },
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "synapse_configure": "%s.NMDA_ratio = 1006.1",
                },
            ]
        },
    },
], indirect=True)
def test_synapse_change_simple_parameters(create_tmp_simulation_config_file):
    """
    Tests simple synapse parameter changes.
    """
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Nd

    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    overrides = {
        ("RingA", "RingA"): {
            "weight": 1001.1,
            "depression_time": 1003.1,
            "facilitation_time": 1002.1,
            "delay": 1004.1
        },
        ("RingA", "RingB"): {
            "hname": "ProbAMPANMDA_EMS",
            "NMDA_ratio": 1006.1
        }}
    for src_pop, src_raw_gid, tgt_pop, tgt_raw_gid in connections:
        src_gid, tgt_gid, edges, selection = utils.get_edge_data(
            nd, src_pop, src_raw_gid, tgt_pop, tgt_raw_gid)
        src_cell, tgt_cell = nd._pc.gid2cell(src_gid), nd._pc.gid2cell(tgt_gid)

        nclist = Nd.cvode.netconlist(src_cell, tgt_cell, "")
        assert len(nclist)
        kwargs = overrides.get((src_pop, tgt_pop), {})
        for nc_id, nc in enumerate(nclist):
            if kwargs:
                # Check that we are not testing when `conductance` is 1.0.
                # Why? Because `weight` is a multiplier of `conductance`:
                # replacement and multiplication have the same effect
                # when `conductance` is 1.0. The behavior cannot be fully
                # tested in that case.
                assert not np.isclose(edges.get_attribute("conductance", selection)[nc_id], 1.0)
            utils.check_netcon(src_gid, nc_id, nc, edges, selection, **kwargs)
            utils.check_synapse(nc.syn(), edges, selection, **kwargs)

    nd.solve(3.0)

    overrides[("RingA", "RingA")]["weight"] = 1005.1
    for src_pop, src_raw_gid, tgt_pop, tgt_raw_gid in connections:
        src_gid, tgt_gid, edges, selection = utils.get_edge_data(
            nd, src_pop, src_raw_gid, tgt_pop, tgt_raw_gid)
        src_cell, tgt_cell = nd._pc.gid2cell(src_gid), nd._pc.gid2cell(tgt_gid)
        nclist = Nd.cvode.netconlist(src_cell, tgt_cell, "")
        assert len(nclist)
        kwargs = overrides.get((src_pop, tgt_pop), {})
        for nc_id, nc in enumerate(nclist):
            # voltage changed from v_init, everything else (i.e. facilitation_time)
            # should be ignored
            utils.check_netcon(src_gid, nc_id, nc, edges, selection,
                               v_init=src_cell.soma[0](0.5).v, **kwargs)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "connection_overrides": [
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "weight": 0,
                }
            ]
        },
    },
], indirect=True)
def test_synapse_without_weight(create_tmp_simulation_config_file):
    """
    Test that 0 weight removes the netcon
    """
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Nd

    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    for src_pop, src_raw_gid, tgt_pop, tgt_raw_gid in connections:
        src_gid, tgt_gid, _edges, _selection = utils.get_edge_data(
            nd, src_pop, src_raw_gid, tgt_pop, tgt_raw_gid)
        src_cell, tgt_cell = nd._pc.gid2cell(src_gid), nd._pc.gid2cell(tgt_gid)

        nclist = Nd.cvode.netconlist(src_cell, tgt_cell, "")
        assert len(nclist) == (0 if src_pop == "RingA" and tgt_pop == "RingB" else 1)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "connection_overrides": [
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "modoverride": "GABAAB",
                }
            ]
        },
    },
], indirect=True)
def test_synapse_modoverride(create_tmp_simulation_config_file):
    """
    Test modoverride
    """
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Nd

    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    overrides = {("RingA", "RingB"): {
        "hname": "ProbGABAAB_EMS",
    }}
    for src_pop, src_raw_gid, tgt_pop, tgt_raw_gid in connections:
        src_gid, tgt_gid, edges, selection = utils.get_edge_data(
            nd, src_pop, src_raw_gid, tgt_pop, tgt_raw_gid)
        src_cell, tgt_cell = nd._pc.gid2cell(src_gid), nd._pc.gid2cell(tgt_gid)

        nclist = Nd.cvode.netconlist(src_cell, tgt_cell, "")
        assert len(nclist)
        kwargs = overrides.get((src_pop, tgt_pop), {})
        for nc_id, nc in enumerate(nclist):
            utils.check_netcon(src_gid, nc_id, nc, edges, selection, **kwargs)
            utils.check_synapse(nc.syn(), edges, selection, **kwargs)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "connection_overrides": [
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "modoverride": "GluSynapse",
                }
            ]
        },
    },
], indirect=True)
def test_gluSynapse_modoverride(create_tmp_simulation_config_file):
    """
    Test modoverride with gluSynapse. It raises an error because of
    missing data in the edge file
    """
    from neurodamus import Neurodamus
    with pytest.raises(AttributeError, match="Missing attribute Use_d_TM in the SONATA edge file"):
        Neurodamus(create_tmp_simulation_config_file, disable_reports=True)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "connection_overrides": [
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "spont_minis": 200
                }
            ]
        }
    },
], indirect=True)
def test_spont_minis_simple(create_tmp_simulation_config_file):
    """Test spont_minis standard behavior.

    This test verifies that a spont_minis is correctly added to
    the simulation, preserving essential properties from existing
    netcons while enforcing expected modifications.

    The test is divided in 2 parts:
    - check that the spont_minis is added, neuron sees it
    and the other netcons are unaffected
    - run a simulation and see spikes at roughly the correct
    firing rate
    """
    from neurodamus import Neurodamus
    from neurodamus.connection import NetConType
    from neurodamus.core import NeuronWrapper as Ndc

    nd = Neurodamus(create_tmp_simulation_config_file)
    # get all the netcons targetting 1001 from neuron directly
    cell = nd._pc.gid2cell(1001)
    nclist = Ndc.cvode.netconlist("", cell, "")
    assert len(nclist) == 3
    # assert that the old netcons are still there
    assert nclist[0].srcgid() == 1
    assert nclist[1].srcgid() == 1002
    # test the spont_minis netcon
    assert nclist[2].srcgid() == -1
    assert nclist[2].weight[4] == int(NetConType.NC_SPONTMINI)
    # weight is set as the weight of the original netcon
    assert nclist[2].weight[0] == nclist[0].weight[0]
    # delay is always set to 0.1. Check connection.SpontMinis.create_on
    assert nclist[2].delay == pytest.approx(0.1)

    voltage_trace = Ndc.Vector()
    cell_ringB = nd.circuits.get_node_manager("RingB").get_cell(1001)
    voltage_trace.record(cell_ringB._cellref.soma[0](0.5)._ref_v)
    Ndc.finitialize()  # reinit for the recordings to be registered
    nd.run()

    utils.check_signal_peaks(voltage_trace, [15,  58, 167, 272, 388], threshold=0.5)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "conditions": {
                "mechanisms": {
                    "ProbAMPANMDA_EMS": {
                        "tau_d_NMDA": 1001.1
                    },
                },
            },
        },
    },
], indirect=True)
def test_override_globals_from_conditions(create_tmp_simulation_config_file):
    """
    Override global synapse variable from the conditions section
    """
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Ndc

    Neurodamus(create_tmp_simulation_config_file, disable_reports=True)

    assert np.isclose(Ndc.h.tau_d_NMDA_ProbAMPANMDA_EMS, 1001.1)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "connection_overrides": [
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "modoverride": "GABAAB",
                    "synapse_configure":
                    "tau_d_NMDA_ProbAMPANMDA_EMS = 1001.1 tau_r_NMDA_ProbAMPANMDA_EMS = 1002.1",
                },
                {
                    "name": "A2B_delayed",
                    "source": "RingA",
                    "target": "RingB",
                    "modoverride": "GABAAB",
                    "synapse_configure": "tau_d_NMDA_ProbAMPANMDA_EMS = 1003.1",
                },
                {
                    "name": "A2A",
                    "source": "RingA",
                    "target": "RingA",
                    "synapse_configure": "tau_d_NMDA_ProbAMPANMDA_EMS = 1005.1",
                },
                {
                    "name": "ZZZ",
                    "source": "RingA",
                    "target": "RingA",
                    "synapse_configure": "tau_d_NMDA_ProbAMPANMDA_EMS = 1006.1",
                },
            ],
            "conditions": {
                "mechanisms": {
                    "ProbAMPANMDA_EMS": {
                        "tau_d_NMDA": 1007.1
                    },
                },
            },
        },
    },
], indirect=True)
def test_override_globals(create_tmp_simulation_config_file):
    """
    Tests whether global synapse parameter overrides take effect as expected.

    Key aspects being tested:
    - The global override ignores synapse type, delay, and order.
    - The order of application might depend on the order in the edges file.
    If equal the order in the `connection_overrides` list.
    - The override in the synapse overrides the one in conditions
    """
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Ndc

    Neurodamus(create_tmp_simulation_config_file, disable_reports=True)

    assert np.isclose(Ndc.h.tau_d_NMDA_ProbAMPANMDA_EMS, 1003.1)
    assert np.isclose(Ndc.h.tau_r_NMDA_ProbAMPANMDA_EMS, 1002.1)
