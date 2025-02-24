import pytest
from pathlib import Path

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_file": "simulation_config_mini.json",
        "src_dir": SIM_DIR / "v5_sonata"
    }
], indirect=True)
def test_crash_test_loading(create_tmp_simulation_config_file):
    from neurodamus import Node
    from neurodamus.cell_distributor import CellDistributor
    from neurodamus.metype import PointCell
    from neuron import nrn

    n = Node(create_tmp_simulation_config_file, {"crash_test": True})
    n.load_targets()
    n.create_cells()

    cell_manager: CellDistributor = n.circuits.get_node_manager("default")
    assert len(cell_manager.cells) == 5
    cell0 = next(iter(cell_manager.cells))
    assert isinstance(cell0, PointCell)
    assert isinstance(cell0.soma, list)
    assert len(cell0.soma) == 1
    assert isinstance(cell0.soma[0], nrn.Section)

    n.create_synapses()
    syn_manager = n.circuits.get_edge_manager("default", "default")
    assert syn_manager.connection_count == 2
