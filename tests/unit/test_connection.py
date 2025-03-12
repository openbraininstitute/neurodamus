""" Test the Connection object """
import numpy as np
import pytest

from neurodamus.core.configuration import ConfigurationError
from neurodamus.node import Node


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic"
        },
    },
], indirect=True)
def test_synapse_location(create_tmp_simulation_config_file):
    """
    Test synapse location recording in Neurodamus.

    This test verifies that synapse locations are correctly recorded in the connections
    as two arrays:
      - `_synapse_sections`: Stores the sections where synapses are located.
      - `_synapse_points_x`: Stores the relative positions of synapses within sections.
    """
    from neurodamus import Neurodamus

    src_pop, tgt_pop = "RingA", "RingB"
    tgid = 1001  # Target cell ID

    # Initialize Neurodamus simulation without reports
    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)

    # Retrieve the target cell
    tgt_cell = nd.circuits.get_node_manager(tgt_pop).get_cell(tgid)

    # Retrieve connections to the target cell from source population
    connlist = list(nd.circuits.get_edge_manager(src_pop, tgt_pop).get_connections(tgid))

    # Ensure there is exactly one connection (other connections may have different sources)
    assert len(connlist) == 1, "Expected exactly one connection to the target cell"
    conn = connlist[0]

    # Retrieve synapse sections and positions
    seclist = list(conn.sections_with_synapses)  # List of synapse-containing sections
    xlist = conn._synapse_points_x  # Synapse positions within sections

    assert len(seclist) == 1, "Expected exactly one synapse section"
    assert len(xlist) == 1, "Expected exactly one synapse position"

    # Validate the synapse is placed at position 0.75 within the section
    assert np.isclose(xlist[0], 0.75), f"Unexpected synapse position: {xlist[0]}"

    # Verify the synapse is in the expected dendritic compartment (compartment 0)
    assert tgt_cell.CCell.dend[0].same(seclist[0][1]), "Synapse not in expected dendrite section"


@pytest.mark.slow
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic",
            "connection_overrides": [
                {
                    "name": "config_ERR",
                    "source": "nodesPopB",
                    "target": "nodesPopB",
                    "synapse_configure": "%s.dummy=1"
                }
            ]
        }
    }
], indirect=True)
def test_config_error(create_tmp_simulation_config_file):

    with pytest.raises(ConfigurationError, match="nodesPopB can't be loaded"):
        n = Node(create_tmp_simulation_config_file)
        n.load_targets()
        n.create_cells()
        n.create_synapses()
        for syn_manager in n._circuits.all_synapse_managers():
            syn_manager.finalize(0, False)
