"""MPI tests of skipping the connections of virtual sources that no replay input can activate.

Each rank decides on the cells it owns, and the connection loop ends in a collective operation,
so every rank must go through it, even those without cells.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from mpi4py import MPI

from tests.replay_circuit import (
    BIO_TO_BIO,
    EXT_TO_BIO,
    N_BIO,
    VIRT_CONNECTIONS,
    ReplayCircuit,
    kept_connections,
    make_replay_circuit,
)

from neurodamus import Neurodamus
from neurodamus.core import NeuronWrapper as Nd
from neurodamus.node import Node

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.fixture
def shared_dir():
    """A directory shared by all the ranks, removed at the end of the test"""
    path = comm.bcast(tempfile.mkdtemp() if rank == 0 else None, root=0)
    yield Path(path)
    comm.Barrier()
    if rank == 0:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch, shared_dir):
    """Replace the conftest fixture, whose per-rank tmp_path is not shared among the ranks"""
    monkeypatch.chdir(shared_dir)


def make_shared_circuit(directory, **kwargs):
    """Create the circuit on rank 0 and return it to every rank"""
    if rank == 0:
        make_replay_circuit(directory, **kwargs)
    comm.Barrier()
    return ReplayCircuit(directory / "circuit_config.json", directory / "node_sets.json", directory)


@pytest.fixture
def circuit_factory(shared_dir):
    return lambda **kwargs: make_shared_circuit(shared_dir, **kwargs)


def shared_spikes_file(circuit, name, population, gids, **kwargs):
    path = circuit.spikes_file(name, population, gids, **kwargs) if rank == 0 else None
    path = comm.bcast(path, root=0)
    comm.Barrier()
    return path


def shared_config(circuit, spikes_files=(), **kwargs):
    path = circuit.write_simulation_config(spikes_files, **kwargs) if rank == 0 else None
    path = comm.bcast(path, root=0)
    comm.Barrier()
    return path


def local_targets(nd):
    manager = nd.circuits.get_node_manager("Bio")
    return sorted(int(gid) for gid in manager.local_nodes.gids(raw_gids=True))


def virt_connections(sources, targets):
    return {
        key: n_syns
        for key, n_syns in VIRT_CONNECTIONS.items()
        if key[0] in sources and key[1] in targets
    }


def connections_by_rank(nd, population):
    """The connections of every rank, by rank.

    All the collective operations of a test must go before its assertions. A rank that fails an
    assertion would otherwise leave the others waiting forever.
    """
    return comm.allgather(kept_connections(nd, population))


def merge(connections_of_ranks):
    """Merge the connections of the ranks. A connection that is in two ranks is an error"""
    merged = {}
    for connections in connections_of_ranks:
        assert merged.keys().isdisjoint(connections.keys())
        merged.update(connections)
    return merged


@pytest.mark.parametrize("replayed", [[0, 3], [1, 2, 4, 5], [], [0, 1, 2, 3, 4, 5]])
@pytest.mark.mpi(ranks=2)
def test_connections_of_all_ranks(circuit_factory, replayed, mpi_ranks):
    circuit = circuit_factory()
    spikes_files = [shared_spikes_file(circuit, "spikes", "Virt", replayed)] if replayed else []
    nd = Neurodamus(shared_config(circuit, spikes_files), disable_reports=True)
    virt, ext, bio = (connections_by_rank(nd, pop) for pop in ("Virt", "Ext", "Bio"))

    assert size == mpi_ranks == 2
    # Round robin: the rank 0 owns the cells 0 and 2, and the rank 1 the cell 1
    assert local_targets(nd) == [[0, 2], [1]][rank]
    # Each rank only has the connections of its cells
    for connections, targets in zip(virt, ([0, 2], [1]), strict=True):
        assert connections == virt_connections(replayed, targets)
    assert merge(virt) == virt_connections(replayed, range(N_BIO))
    assert merge(ext) == {}
    assert set(merge(bio)) == set(BIO_TO_BIO)


@pytest.mark.mpi(ranks=[4, 5])
def test_ranks_without_cells_do_not_block(circuit_factory, mpi_ranks):
    """With more ranks than cells, some ranks are empty and still take part in the collectives"""
    circuit = circuit_factory()
    spikes_file = shared_spikes_file(circuit, "spikes", "Virt", [3, 4])
    nd = Neurodamus(shared_config(circuit, [spikes_file]), disable_reports=True)
    virt = connections_by_rank(nd, "Virt")
    targets = comm.allgather(local_targets(nd))

    assert size == mpi_ranks
    assert sum(1 for rank_targets in targets if not rank_targets) == mpi_ranks - N_BIO
    for rank_targets, connections in zip(targets, virt, strict=True):
        assert connections == virt_connections([3, 4], rank_targets)
    assert merge(virt) == virt_connections([3, 4], range(N_BIO))


@pytest.mark.mpi(ranks=3)
def test_one_cell_per_rank_with_minis_on_a_single_rank(circuit_factory, mpi_ranks):
    """Only the rank of the cell with minis keeps all of its connections"""
    circuit = circuit_factory(exc_minis=[0.0, 4.0, 0.0], inh_minis=[0.0, 0.0, 0.0])
    spikes_file = shared_spikes_file(circuit, "spikes", "Virt", [3])
    nd = Neurodamus(shared_config(circuit, [spikes_file]), disable_reports=True)
    virt = connections_by_rank(nd, "Virt")

    assert local_targets(nd) == [rank]
    # The cell 1 has minis and keeps both its connections. The other cells only the ones from the
    # replayed source 3, and none of them has any
    assert virt[0] == virt[2] == {}
    assert virt[1] == virt_connections([0, 1, 2, 3, 4, 5], [1])
    assert merge(virt) == virt_connections([0, 1, 2, 3, 4, 5], [1])


@pytest.mark.mpi(ranks=2)
def test_no_replay_input(circuit_factory, mpi_ranks):
    circuit = circuit_factory()
    nd = Neurodamus(shared_config(circuit), disable_reports=True)
    virt, ext, bio = (connections_by_rank(nd, pop) for pop in ("Virt", "Ext", "Bio"))

    assert merge(virt) == {}
    assert merge(ext) == {}
    assert set(merge(bio)) == set(BIO_TO_BIO)


@pytest.mark.mpi(ranks=2)
def test_dry_run_counts_every_synapse(circuit_factory, mpi_ranks):
    circuit = circuit_factory()
    spikes_file = shared_spikes_file(circuit, "spikes", "Virt", [3])
    config = shared_config(circuit, [spikes_file])
    nd = Neurodamus(config, dry_run=True, num_target_ranks=2)
    nd.run()
    # Each rank counts the synapses of its own cells
    n_synapses = comm.allreduce(sum(nd._dry_run_stats.synapse_counts.values()))
    n_total = sum(VIRT_CONNECTIONS.values()) + len(EXT_TO_BIO) + len(BIO_TO_BIO)
    assert n_synapses == pytest.approx(n_total, rel=0.05)


def run_and_record(nd):
    manager = nd.circuits.get_node_manager("Bio")
    voltages = {}
    for gid in manager.local_nodes.gids(raw_gids=False):
        voltage = Nd.Vector()
        voltage.record(manager.get_cell(int(gid))._cellref.soma[0](0.5)._ref_v)
        voltages[int(gid)] = voltage
    Nd.finitialize()
    nd.run(cleanup=False)
    return {gid: voltage.as_numpy().copy() for gid, voltage in voltages.items()}


@pytest.mark.parametrize("replayed", [[0, 3], [1, 2, 4]])
@pytest.mark.mpi(ranks=2)
def test_simulation_does_not_change(circuit_factory, monkeypatch, replayed, mpi_ranks):
    """The voltages of all the cells are the same as without skipping any connection"""
    circuit = circuit_factory()
    spikes_file = shared_spikes_file(
        circuit, "spikes", "Virt", replayed, times=(2.0, 5.0, 9.0, 15.0)
    )
    config = shared_config(circuit, [spikes_file])
    skipper = Node._can_skip_unreplayed_connections

    results = {}
    for skip in (True, False):
        monkeypatch.setattr(
            Node,
            "_can_skip_unreplayed_connections",
            lambda *args, skip=skip: skip and skipper(*args),
        )
        nd = Neurodamus(config, disable_reports=True)
        voltages = run_and_record(nd)
        results[skip] = (voltages, connections_by_rank(nd, "Virt"))
        nd.clear_model()
        Nd.t = 0.0
    spread = comm.allreduce(max(np.ptp(v) for v in results[False][0].values()), op=MPI.MAX)

    assert merge(results[True][1]) == virt_connections(replayed, range(N_BIO))
    assert merge(results[False][1]) == VIRT_CONNECTIONS
    for gid, voltage in results[False][0].items():
        npt.assert_array_equal(results[True][0][gid], voltage)
    # The replay has an effect on the voltage
    assert spread > 0.5
