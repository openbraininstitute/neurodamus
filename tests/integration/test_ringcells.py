import os
from pathlib import Path


SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
REF_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest" / "reference"
CONFIG_FILE = str(SIM_DIR / "simulation_config.json")


def test_dumpcellstates():
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd
    n = Neurodamus(CONFIG_FILE, disable_reports=True)
    from neurodamus.utils.dump_cellstate import dump_cellstate
    test_gids = [1, 1001]
    for gid in test_gids:
        dump_cellstate(n._pc, Nd.cvode, gid)
        outputfile = "cellstate_" + str(gid) + ".json"
        reference = REF_DIR / outputfile
        compare_json_files(outputfile, str(reference))


def compare_json_files(res_file, ref_file):
    assert os.path.isfile(res_file)
    assert os.path.isfile(ref_file)
    import json
    with open(res_file) as f_res:
        result = json.load(f_res)
    with open(ref_file) as f_ref:
        reference = json.load(f_ref)
    assert result  == reference
