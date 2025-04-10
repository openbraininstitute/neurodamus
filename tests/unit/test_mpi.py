import os
import pytest
from pathlib import Path
# !! NOTE: Please don't import neron/neurodamus at module level
# pytest weird discovery system will trigger Neuron init and open can of worms


@pytest.fixture(scope="session")
def morphologies_root(rootdir):
    return Path(rootdir) / "tests" / "sample_data" / "morphology"


@pytest.fixture
def Cell(rootdir):
    os.environ.setdefault("HOC_LIBRARY_PATH", str(rootdir / "neurodamus" / "data" / "hoc"))
    from neurodamus.core.cell import Cell
    return Cell


@pytest.mark.parametrize(
    "morphology_path",
    ["C060114A7.asc", "C060114A7.h5", "merged_container.h5/C060114A7.h5"]
)
def test_load_cell(morphologies_root, morphology_path, Cell):
    c = Cell(1, str(morphologies_root / morphology_path))
    assert len(c.all) == 325
    assert c.axons[4].name().endswith(".axon[4]")
    assert c.soma.L == pytest.approx(26.11, abs=0.01)
    from neurodamus.core import Neuron
    Neuron.load_hoc("neurodamus")



def test_basic_system():
    from neurodamus.core import Neuron
    from neurodamus.core.cell import Cell
    from neurodamus.core.stimuli import CurrentSource

    # CurrentSource invokes neuron.load_hoc("neurodamus") which loads "cell.hoc"
    # with the same hoc template name as Cell.py.
    # Neuron 9 throws error during loading "Cell :a template cannot be redefined".
    # To avoid it, we load the cell.hoc file first

    Neuron.load_hoc("neurodamus")

# def test_basic_system2():
#     from neurodamus.core import Neuron
#     from neurodamus.core.cell import Cell
#     from neurodamus.core.stimuli import CurrentSource

#     # CurrentSource invokes neuron.load_hoc("neurodamus") which loads "cell.hoc"
#     # with the same hoc template name as Cell.py.
#     # Neuron 9 throws error during loading "Cell :a template cannot be redefined".
#     # To avoid it, we load the cell.hoc file first

#     Neuron.load_hoc("neurodamus")


# def test_morphio_read(morphologies_root, Cell):
#     c = Cell(1, str(morphologies_root / "simple.h5"))
#     Cell.show_topology()
#     assert len(c.all) == 7
#     assert len(list(c.h.basal)) == 3
#     assert len(list(c.h.axonal)) == 3


# def test_create_cell(Cell):
#     builder = Cell.Builder
#     c = (builder
#          .add_soma(1)
#          .add_dendrite("dend1", 2, 5)
#          .attach(builder.DendriteSection("dend2", 3, 2).add("sub2_dend", 4, 2))
#          .add_axon("axon1", 2, 3)
#          .create())

#     Cell.show_topology()
#     assert len(c.all) == 5
#     assert len(list(c.h.basal)) == 3
#     assert len(c._dend) == 3
#     assert len(c._axon) == 1


# def test_create_cell_2(Cell):
#     c = (Cell.Builder
#          .add_soma(1)
#          .add_dendrite("dend1", 2, 5)
#          .append_axon("ax1", 3, 2).append("ax1_2", 4, 2).append("ax1_3", 3, 3)
#          .create())

#     Cell.show_topology()
#     assert len(c.all) == 5
#     assert len(list(c.h.basal)) == 1
#     assert len(c._dend) == 1
#     assert len(c._axon) == 3


# def test_create_cell_3(Cell):
#     Dend = Cell.Builder.DendriteSection
#     c = (Cell.Builder
#          .add_soma(1)
#          .add_dendrite("dend1", 2, 5)
#          .attach(Dend("dend2", 3, 2)
#                  .append("sub2_dend", 4, 2)
#                  .get_root())
#          .create())

#     Cell.show_topology()
#     assert len(c.all) == 4
#     assert len(list(c.h.basal)) == 3
#     assert len(c._dend) == 3
#     assert c._axon is None
#     assert len(c.axons) == 0






##########################

# import pytest
# from neurodamus import Neurodamus

# # @pytest.mark.mpi(ranks=2)
# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     },
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#         }
#     }
# ], indirect=True)
# def test_mpi_rank_communication(create_tmp_simulation_config_file, mpi_ranks):
#     nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
#     nd.run()

# # @pytest.mark.mpi(ranks=2)
# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     },
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#         }
#     }
# ], indirect=True)
# def test_mpi_rank_communication2(create_tmp_simulation_config_file, mpi_ranks):
#     nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
#     nd.run()

    # print("AAAAAAA", nd._pc.nhost())
    # assert nd._pc.nhost() == 1
    # rank = comm.rank
    # size = comm.size

    # assert size == 2, f"Expected 2 ranks, got {size}"

    # if rank == 0:
    #     comm.send("Hello from 0", dest=1, tag=123)
    # elif rank == 1:
    #     msg = comm.recv(source=0, tag=123)
    #     assert msg == "Hello from 0"

# import pytest
# from mpi4py import MPI


# @pytest.mark.mpi(ranks=[2])

# def test_without_mpi(mpi_ranks, create_tmp_simulation_config_file, comm):

#     # """Simple passing test"""
#     
#     nd.run()
    # print(mpi_ranks, comm.rank)

    # rank = comm.rank
    # size = comm.size

    # assert size >= 2, f"Test requires at least 2 ranks, but got {size}"

    # if rank == 0:
    #     data = "Hello from rank 0"
    #     comm.send(data, dest=1, tag=42)
    # elif rank == 1:
    #     data = comm.recv(source=0, tag=42)
    #     assert data == "Hello from rank 0", f"Rank 1 received incorrect data: {data}"

# @pytest.mark.mpi(ranks=2)
# def test_with_mpi(mpi_ranks):  # pylint: disable=unused-argument
#     """Simple passing test"""
#     assert True  # replace with actual test code

# @pytest.mark.mpi(ranks=2)
# def test_one_failing_rank(mpi_ranks, comm):  # pylint: disable=unused-argument
#     """In case of just one process failing an assert, the test counts
#     as failed and the outputs are gathered from the processes."""

#     assert comm.rank <= 1

# from mpi4py import MPI
# from neurodamus import Neurodamus


# comm = MPI.COMM_WORLD

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     },
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#         }
#     }
# ], indirect=True)
# @pytest.mark.mpi(minsize=2)
# def test_no_mpi():
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     assert comm.size >= 2
    # nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    # nd.run()

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     },
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#         }
#     }
# ], indirect=True)
# @pytest.mark.mpi(minsize=2)
# def test_mpi(create_tmp_simulation_config_file):
#     pass
#     # nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
#     # nd.run()

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     },
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#         }
#     }
# ], indirect=True)
# @pytest.mark.mpi(minsize=3)
# def test_mpi2(create_tmp_simulation_config_file):
#     pass
#     # nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
#     # nd.run()

# from neurodamus import Neurodamus

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     }
# ], indirect=True)
# @pytest.mark.mpi(min_size=2)
# def test_mpi():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     assert size >= 2
#     assert size <= 3
    # nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    # nd.run()
