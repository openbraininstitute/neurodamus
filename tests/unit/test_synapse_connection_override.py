
# from pathlib import Path

# import numpy as np
# import pytest


# from neurodamus.core.configuration import SimConfig
# from tests import utils

# SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
# REF_DIR = SIM_DIR / "reference"
# CONFIG_FILE = str(SIM_DIR / "simulation_config.json")


# def inspect(v):
#     print(v, type(v))
#     for i in dir(v):
#         if i.startswith("_"):
#             continue
#         print(i)


# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#             "node_set": "Mosaic",
#             "connection_overrides": [
#                 {
#                     "name": "A2B",
#                     "source": "RingA",
#                     "target": "RingB",
#                     "weight": 1111
#                 }
#             ]
#         },
#     },
# ], indirect=True)
# def test_synapse_change_parameter(create_tmp_simulation_config_file):
#     from neurodamus import Neurodamus
#     from neurodamus.core import NeurodamusCore as Nd

#     n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)

#     sgid, tgid, edges, selection = utils.get_gid_edges_selection(n, "RingA", 1, "RingB", 1)
#     nclist = Nd.cvode.netconlist(n._pc.gid2cell(sgid), n._pc.gid2cell(tgid), "")
#     # assert len(nclist)
#     # for nc_id, nc in enumerate(nclist):
#     #     utils.check_netcon(sgid, nc_id, nc, edges, selection, miao=1111)
