import json
import os
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile


SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
REF_DIR = SIM_DIR / "reference"
CONFIG_FILE = str(SIM_DIR / "simulation_config.json")


def compare_json_files(res_file, ref_file):
    assert os.path.isfile(res_file)
    assert os.path.isfile(ref_file)
    import json
    with open(res_file) as f_res:
        result = json.load(f_res)
    with open(ref_file) as f_ref:
        reference = json.load(f_ref)
    assert result  == reference


@pytest.fixture
def simconfig_nodeset(ringtest_baseconfig, extra_config):
    ringtest_baseconfig.update(extra_config)

    with NamedTemporaryFile("w", suffix='.json', delete=False) as config_file:
        json.dump(ringtest_baseconfig, config_file)

    yield config_file
    os.unlink(config_file.name)


@pytest.mark.parametrize("extra_config", [{"node_set": "Mosaic"}])
def test_dump_allcells(simconfig_nodeset):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd
    n = Neurodamus(simconfig_nodeset.name, disable_reports=True)
    from neurodamus.utils.dump_cellstate import dump_cellstate
    test_gids = [1, 1001]
    for gid in test_gids:
        dump_cellstate(n._pc, Nd.cvode, gid)
        outputfile = "cellstate_" + str(gid) + ".json"
        reference = REF_DIR / outputfile
        compare_json_files(outputfile, str(reference))


# @pytest.mark.parametrize("extra_config", [{"node_set": "RingB"}])
# def test_dump_RingB():
