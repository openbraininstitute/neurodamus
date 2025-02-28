
from pathlib import Path

import pytest
import numpy as np

from tests import utils

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
REF_DIR = SIM_DIR / "reference"
CONFIG_FILE = str(SIM_DIR / "simulation_config.json")


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

    Note: The weight acts as a conductance multiplier. Avoid using `RingA -> RingB`
    NetCons, as their base conductance is `1.0`. In this case, multiplication and
    replacement result in the same effect, making it impossible to
    distinguish between them.
    """
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    overrides = {"weight": 10001.1,
                 "depression_time": 10003.1,
                 "facilitation_time": 10002.1,
                 "delay": 10005.1
                 }
    for conn in connections:
        sgid, tgid, edges, selection = utils.get_gid_edges_selection(n, *conn)
        scell, tcell = n._pc.gid2cell(sgid), n._pc.gid2cell(tgid)

        nclist = Nd.cvode.netconlist(scell, tcell, "")
        assert len(nclist)
        kwargs = overrides if (conn[0] == "RingA" and conn[2] == "RingA") else {}
        for nc_id, nc in enumerate(nclist):
            if kwargs:
                # Check that we are not testing when `conductance` is 1.0.
                # Why? Because `weight` is a multiplier of `conductance`:
                # replacement and multiplication have the same effect
                # when `conductance` is 1.0. The behavior cannot be
                # fully tested in that case.
                assert np.isclose(edges.get_attribute("conductance", selection)[nc_id], 1.0)
            utils.check_netcon(sgid, nc_id, nc, edges, selection, **kwargs)
            utils.check_synapse(nc.syn(), edges, selection, **kwargs)

    n.solve(3.0)

    overrides["weight"] = 10004.1
    for conn in connections:
        sgid, tgid, edges, selection = utils.get_gid_edges_selection(n, *conn)
        scell, tcell = n._pc.gid2cell(sgid), n._pc.gid2cell(tgid)
        nclist = Nd.cvode.netconlist(scell, tcell, "")
        assert len(nclist)
        kwargs = overrides if (conn[0] == "RingA" and conn[2] == "RingA") else {}
        for nc_id, nc in enumerate(nclist):
            # voltage changed from v_init, everything else (i.e. facilitation_time)
            # should be ignored
            utils.check_netcon(sgid, nc_id, nc, edges, selection,
                               v_init=scell.soma[0](0.5).v, **kwargs)


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

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    for conn in connections:
        sgid, tgid, _, _ = utils.get_gid_edges_selection(n, *conn)
        scell, tcell = n._pc.gid2cell(sgid), n._pc.gid2cell(tgid)

        nclist = Nd.cvode.netconlist(scell, tcell, "")
        assert len(nclist) == (0 if conn[0] == "RingA" and conn[2] == "RingB" else 1)


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

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    overrides = {
        "hname": "ProbAMPANMDA_EMS",
    }
    for conn in connections:
        sgid, tgid, edges, selection = utils.get_gid_edges_selection(n, *conn)
        scell, tcell = n._pc.gid2cell(sgid), n._pc.gid2cell(tgid)

        nclist = Nd.cvode.netconlist(scell, tcell, "")
        assert len(nclist)
        kwargs = overrides if conn[0] == "RingA" and conn[2] == "RingB" else {}
        for nc_id, nc in enumerate(nclist):
            utils.check_netcon(sgid, nc_id, nc, edges, selection, **kwargs)
            utils.check_synapse(nc.syn(), edges, selection, **kwargs)
