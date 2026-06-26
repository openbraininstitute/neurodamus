from pathlib import Path

import numpy.testing as npt
import pytest

from tests import utils
from tests.conftest import ALLEN_V1_DIR

from neurodamus import Neurodamus
from neurodamus.core import MPI
from neurodamus.core.configuration import ConfigurationError
from neurodamus.utils.dump_cellstate import dump_cellstate

rank = MPI.rank


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "src_dir": ALLEN_V1_DIR,
        }
    ],
    indirect=True,
)
@pytest.mark.mpi(ranks=2)
def test_cell_states(capsys, create_tmp_simulation_config_file, mpi_ranks):
    from neurodamus.core import NeuronWrapper as Nd

    assert mpi_ranks == 2
    n = Neurodamus(create_tmp_simulation_config_file)

    # 1. check warning msg about no BBP syn models
    captured = capsys.readouterr()
    if rank == 0:
        assert "[WARNING] Could not find BBP synapse models from NRNMECH_LIB_PATH" in captured.out

    # 2. compare the cell state of the biophyiscal cell gid=0 in rank 0
    if rank == 0:
        tgid = 0
        outputfile = "allen_v1_cellstate_" + str(tgid) + ".json"
        dump_cellstate(n._pc, Nd.cvode, tgid, outputfile)
        reference = ALLEN_V1_DIR / "reference" / outputfile
        utils.compare_json_files(Path(outputfile), reference)

    # 3. check netcons attach to the artificial point cell gid=1000
    # because point cell is artificial, we can't access it by netcon.postcelllist() [Neuron doc]
    # or cvode.netconlist
    # but via our python object

    if rank == 0:
        # netcon 0->1000
        tgid = 1000
        sgid = 0
        edges_ab = n.circuits.get_edge_manager("RingA", "RingB")
        nc_list = [nc for conn in edges_ab.get_connections(tgid, sgid) for nc in conn._netcons]
        assert len(nc_list) == 1
        netcon = nc_list[0]
        assert netcon.precell().hname() == "TestCell[0]"
        assert netcon.srcgid() == 0
        npt.assert_allclose(netcon.weight[0], 500)
        npt.assert_allclose(netcon.threshold, -30)
        npt.assert_allclose(netcon.delay, 3)

        # all netcons with the same target to the point cell
        syns = netcon.synlist()
        assert syns[0] == netcon  # the current netcon 0->1000
        assert syns[1].srcgid() == 1002  # netcon in 1002->1000
        assert syns[2].srcgid() == -1  # netcon for synapse replay
        assert syns[2].pre().hname() == "VecStim[0]"
        assert syns[2].weight[0] == 500


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"src_dir": ALLEN_V1_DIR, "extra_config": {"conditions": {"randomize_gaba_rise_time": True}}}],
    indirect=True,
)
@pytest.mark.mpi(ranks=2)
def test_errorhandling(create_tmp_simulation_config_file, mpi_ranks):
    """
    The Allen v1 circuit does not contain the GABAAB synapse model.
    If the simulation config file contains "randomize_gaba_rise_time": True, an exception is raised
    """
    assert mpi_ranks == 2

    with pytest.raises(
        ConfigurationError,
        match=r"Cannot enable randomize_gaba_rise_time, missing ProbGABAAB_EMS.mod",
    ):
        Neurodamus(create_tmp_simulation_config_file)
