"""A small circuit to test the replay of virtual projections.

Populations:
    Bio:  3 biophysical cells, optionally with spontaneous minis
    Virt: 6 virtual sources projecting to Bio. Several sources per target and several synapses
          per connection
    Ext:  3 virtual sources projecting to Bio. Never replayed in the tests
    Bio -> Bio: internal connectivity between the real cells
"""

import itertools as it
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from tests.conftest import RINGTEST_DIR
from tests.simulations.utils import Edges, make_edges, make_nodes

# (source, target) pairs, one per synapse, sorted by target and then by source. The repeated
# pairs are connections with several synapses
VIRT_TO_BIO = [
    (0, 0), (0, 0), (1, 0), (2, 0),
    (0, 1), (3, 1), (3, 1), (3, 1),
    (1, 2), (4, 2), (5, 2),
]  # fmt: skip
EXT_TO_BIO = [(0, 0), (1, 1), (2, 2)]
BIO_TO_BIO = [(0, 1), (1, 2), (2, 0)]

# The number of synapses per connection, by (source, target)
VIRT_CONNECTIONS = {
    (0, 0): 2, (1, 0): 1, (2, 0): 1,
    (0, 1): 1, (3, 1): 3,
    (1, 2): 1, (4, 2): 1, (5, 2): 1,
}  # fmt: skip

N_VIRT = 6
N_EXT = 3
N_BIO = 3


@dataclass
class ReplayCircuit:
    circuit_config: Path
    node_sets_file: Path
    directory: Path

    def spikes_file(self, name, population, gids, times=(5.0,)):
        """Write a SONATA spikes file where each gid spikes at each of the times"""
        path = self.directory / f"{name}.h5"
        node_ids = np.repeat(np.asarray(gids, dtype="uint64"), len(times))
        timestamps = np.tile(np.asarray(times, dtype="float64"), len(gids))
        with h5py.File(path, "w") as h5:
            group = h5.create_group(f"spikes/{population}")
            group.create_dataset("node_ids", data=node_ids)
            group.create_dataset("timestamps", data=timestamps)
            group["timestamps"].attrs["units"] = "ms"
        return str(path)

    def simulation_config(self, spikes_files=(), **extra_config):
        """The simulation config. Each spikes file in `spikes_files` is a replay input"""
        config = {
            "network": str(self.circuit_config),
            "node_sets_file": str(self.node_sets_file),
            "target_simulator": "NEURON",
            "run": {"random_seed": 1122, "dt": 0.1, "tstop": 30},
            "node_set": "Bio",
            "conditions": {"celsius": 35, "v_init": -65},
            "inputs": {
                f"replay_{i}": {
                    "module": "synapse_replay",
                    "input_type": "spikes",
                    "spike_file": spikes_file,
                    "delay": 0,
                    "duration": 30,
                    "node_set": "Bio",
                }
                for i, spikes_file in enumerate(spikes_files)
            },
        }
        config.update(extra_config)
        return config

    def write_simulation_config(self, spikes_files=(), name="simulation_config.json", **extra):
        path = self.directory / name
        path.write_text(json.dumps(self.simulation_config(spikes_files, **extra)))
        return str(path)


def _edge_attributes(count):
    return {
        "edge_type_id": -1,
        "conductance": it.cycle([2.0, 3.0, 4.0]),
        "decay_time": 2.0,
        "delay": 0.5,
        "depression_time": 100.0,
        "facilitation_time": 100.0,
        "u_syn": 0.5,
        "afferent_section_id": 1,
        "afferent_section_pos": it.cycle([0.25, 0.5, 0.75]),
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": [104] * count,
    }


def _virtual_nodes(path, name, count):
    make_nodes(filename=path, name=name, count=count, wanted_attributes={"node_type_id": -1})


def make_replay_circuit(directory, exc_minis=None, inh_minis=None):
    """Create the circuit in `directory`

    Args:
        directory: Where to write the circuit
        exc_minis: Optional exc_mini_frequency of each Bio cell
        inh_minis: Optional inh_mini_frequency of each Bio cell. Needs exc_minis too
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    make_nodes(
        filename=str(directory / "nodes_bio.h5"),
        name="Bio",
        count=N_BIO,
        wanted_attributes={
            "node_type_id": -1,
            "model_template": "hoc:TestCell",
            "model_type": "biophysical",
            "mtype": ["MTYPE0", "MTYPE1", "MTYPE2"],
            "etype": ["ETYPE0", "ETYPE1", "ETYPE2"],
            "x": it.count(0),
            "y": it.count(1),
            "z": it.count(2),
            "morphology": "cell_small",
        },
    )
    if exc_minis is not None:
        with h5py.File(directory / "nodes_bio.h5", "a") as h5:
            group = h5["nodes/Bio/0"]
            group.create_dataset("exc_mini_frequency", data=np.asarray(exc_minis, dtype="f8"))
            group.create_dataset(
                "inh_mini_frequency", data=np.asarray(inh_minis or [0.0] * N_BIO, dtype="f8")
            )
    _virtual_nodes(str(directory / "nodes_virt.h5"), "Virt", N_VIRT)
    _virtual_nodes(str(directory / "nodes_ext.h5"), "Ext", N_EXT)

    for fname, src, tgt, pairs in (
        ("edges_virt.h5", "Virt", "Bio", VIRT_TO_BIO),
        ("edges_ext.h5", "Ext", "Bio", EXT_TO_BIO),
        ("edges_bio.h5", "Bio", "Bio", BIO_TO_BIO),
    ):
        edges = Edges(src, tgt, "chemical", pairs)
        make_edges(str(directory / fname), edges, _edge_attributes(len(pairs)))

    circuit_config = {
        "version": 2,
        "networks": {
            "nodes": [
                {
                    "nodes_file": str(directory / "nodes_bio.h5"),
                    "populations": {
                        "Bio": {
                            "type": "biophysical",
                            "morphologies_dir": str(RINGTEST_DIR / "morphologies/swc"),
                            "biophysical_neuron_models_dir": str(RINGTEST_DIR / "hoc"),
                            "alternate_morphologies": {
                                "neurolucida-asc": str(RINGTEST_DIR / "morphologies/asc")
                            },
                        }
                    },
                },
                {
                    "nodes_file": str(directory / "nodes_virt.h5"),
                    "populations": {"Virt": {"type": "virtual"}},
                },
                {
                    "nodes_file": str(directory / "nodes_ext.h5"),
                    "populations": {"Ext": {"type": "virtual"}},
                },
            ],
            "edges": [
                {
                    "edges_file": str(directory / "edges_bio.h5"),
                    "populations": {"Bio__Bio__chemical": {"type": "chemical"}},
                },
                {
                    "edges_file": str(directory / "edges_virt.h5"),
                    "populations": {"Virt__Bio__chemical": {"type": "chemical"}},
                },
                {
                    "edges_file": str(directory / "edges_ext.h5"),
                    "populations": {"Ext__Bio__chemical": {"type": "chemical"}},
                },
            ],
        },
    }
    circuit_config_file = directory / "circuit_config.json"
    circuit_config_file.write_text(json.dumps(circuit_config))
    node_sets_file = directory / "node_sets.json"
    node_sets_file.write_text(
        json.dumps(
            {
                "Bio": {"population": "Bio"},
                "Virt": {"population": "Virt"},
                "Ext": {"population": "Ext"},
            }
        )
    )
    return ReplayCircuit(circuit_config_file, node_sets_file, directory)


def kept_connections(nd, src_pop, dst_pop="Bio"):
    """The connections created by the edge manager as {(raw source, raw target): n synapses}"""
    manager = nd.circuits.get_edge_manager(src_pop, dst_pop)
    if manager is None:
        return {}
    src_offset, dst_offset = manager.src_pop_offset, manager.target_pop_offset
    return {
        (conn.sgid - src_offset, conn.tgid - dst_offset): len(conn._synapse_ids)
        for conn in manager.all_connections()
    }
