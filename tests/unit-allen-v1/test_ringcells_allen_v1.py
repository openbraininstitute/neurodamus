from pathlib import Path

import numpy.testing as npt

from tests import utils

from neurodamus import Neurodamus
from neurodamus.utils.dump_cellstate import dump_cellstate

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest_allen_v1"


def test_cell_states(capsys):
    from neurodamus.core import NeuronWrapper as Nd

    sim_conf = str(SIM_DIR / "simulation_config.json")
    n = Neurodamus(sim_conf)

    # 1. check warning msg about no BBP syn models
    captured = capsys.readouterr()
    assert "[WARNING] Could not find BBP synapse models from NRNMECH_LIB_PATH" in captured.out

    # 2. compare the cell state of the biophyiscal cell gid=0
    tgid = 0
    outputfile = "allen_v1_cellstate_" + str(tgid) + ".json"
    dump_cellstate(n._pc, Nd.cvode, tgid, outputfile)
    reference = SIM_DIR / "reference" / outputfile
    utils.compare_json_files(Path(outputfile), reference)

    # 3. check netcons attach to the artificial point cell gid=1000
    # because point cell is artificial, we can't access it by netcon.postcelllist() [Neuron doc]
    # or cvode.netconlist
    # but via our python object

    # netcon 0->1000
    tgid = 1000
    sgid = 0
    edges_ab = n.circuits.get_edge_manager("RingA", "RingB")
    nc_list = [nc for conn in edges_ab.get_connections(tgid, sgid) for nc in conn._netcons]
    assert len(nc_list) == 1
    netcon = nc_list[0]
    assert netcon.precell().hname() == "TestCell[0]"
    assert netcon.srcgid() == 0
    npt.assert_allclose(netcon.weight[0], 100)
    npt.assert_allclose(netcon.threshold, -30)
    npt.assert_allclose(netcon.delay, 3)

    # all netcons with the same target to the point cell
    syns = netcon.synlist()
    assert syns[0] == netcon  # the current netcon 0->1000
    assert syns[1].srcgid() == 1002  # netcon in 1002->1000
    assert syns[2].srcgid() == -1  # netcon for synapse replay
    assert syns[2].pre().hname() == "VecStim[0]"
