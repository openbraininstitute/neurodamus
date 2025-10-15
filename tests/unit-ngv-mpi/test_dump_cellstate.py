
import pytest

from neurodamus import Neurodamus
from tests.conftest import NGV_DIR
from neurodamus.utils.dump_cellstate import dump_cellstate
from pathlib import Path



@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(NGV_DIR),
        "simconfig_file": "simulation_config.json"
    }
], indirect=True)
@pytest.mark.mpi(ranks=1)
def test_simple_dump(create_tmp_simulation_config_file, mpi_ranks):
    """
    Test cell_dumpstate with a neuron and an astrocyte
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    astro_ids = list(n.circuits.get_node_manager("AstrocyteA").gid2cell.keys())

    for astro_id in astro_ids:
        outputfile = Path("cellstate_" + str(astro_id) + ".json")
        dump_cellstate(n._pc, Nd.cvode, astro_id, outputfile)
        assert outputfile.exists(), f"Missing dump file: {outputfile}"




