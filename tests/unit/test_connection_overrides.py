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
                    "weight": 10001.1,
                    "synapse_configure": "%s.Fac = 10002.1 %s.Dep = 10003.1",
                    "delay": 0.0,
                    "synapse_delay_override": 10005.1
                },
                {
                    "name": "A2A_delayed",
                    "source": "RingA",
                    "target": "RingA",
                    "weight": 10004.1,
                    "synapse_configure": "%s.Fac = 10005.1",
                    "delay": 1.0
                }
            ]
        },
    },
], indirect=True)
def test_synapse_change_simple_parameters(create_tmp_simulation_config_file):
    """
    Tests simple synapse parameter changes.
    """
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd

    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    overrides = {
        ("RingA", "RingA"): {
            "weight": 10001.1,
            "depression_time": 10003.1,
            "facilitation_time": 10002.1,
            "delay": 10005.1
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

    overrides[("RingA", "RingA")]["weight"] = 10004.1
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
    from neurodamus.core import NeurodamusCore as Nd

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
                    "modoverride": "AMPANMDA"
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
    from neurodamus.core import NeurodamusCore as Nd

    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    overrides = {("RingA", "RingB"): {
        "hname": "ProbAMPANMDA_EMS",
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
                    "spont_minis": 200,
                    "synapse_configure": "%s.verboseLevel = 1"
                }
            ]
        }
    },
], indirect=True)
def test_spont_minis_simple(create_tmp_simulation_config_file):
    """Test that spont_mini fires with roughly """
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Ndc

    cell_id = 1001

    nd = Neurodamus(create_tmp_simulation_config_file)
    manager = nd.circuits.get_node_manager("RingB")
    cell_ringB = manager.get_cell(cell_id)
    voltage_trace = Ndc.Vector()
    voltage_trace.record(cell_ringB._cellref.soma[0](0.5)._ref_v)
    Ndc.finitialize()  # reinit for the recordings to be registered
    nd.run()

    # with threshold=1.0 it does not get the last peak
    utils.check_signal_peaks(voltage_trace, [ 12,  55, 122, 164, 269, 303, 385], threshold=0.5)
