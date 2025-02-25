#!/bin/env python
# /// script
# dependencies = ['h5py', 'libsonata', 'numpy']
# ///
# the above allows one to run `uv run create_data.py` without a virtualenv
import itertools as it
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import h5py
import libsonata
import numpy as np


@dataclass
class SonataAttribute:
    name: str
    type: type
    prefix: bool


NODE_TYPES = [
    SonataAttribute("node_type_id", type=int, prefix=False),
    SonataAttribute("model_template", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("mtype", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("x", type=np.float32, prefix=True),
    SonataAttribute("y", type=np.float32, prefix=True),
    SonataAttribute("z", type=np.float32, prefix=True),
    SonataAttribute("morphology", type=h5py.string_dtype(), prefix=True),
    ]
NODE_TYPES = {attr.name: attr for attr in NODE_TYPES}

EDGE_TYPES = [
    SonataAttribute("edge_type_id", type=int, prefix=False),
    SonataAttribute("conductance", type=np.float32, prefix=True),
    SonataAttribute("decay_time", type=np.float32, prefix=True),
    SonataAttribute("delay", type=np.float32, prefix=True),
    SonataAttribute("depression_time", type=np.float32, prefix=True),
    SonataAttribute("facilitation_time", type=np.float32, prefix=True),
    SonataAttribute("u_syn", type=np.float32, prefix=True),

    SonataAttribute("afferent_section_id", type=int, prefix=True),
    SonataAttribute("afferent_section_pos", type=np.float32, prefix=True),
    SonataAttribute("afferent_segment_id", type=int, prefix=True),
    SonataAttribute("afferent_segment_offset", type=int, prefix=True),

    SonataAttribute("n_rrp_vesicles", type=int, prefix=True),
    SonataAttribute("syn_type_id", type=int, prefix=True),
    ]
EDGE_TYPES = {attr.name: attr for attr in EDGE_TYPES}


@dataclass
class Edges:
    src: str
    tgt: str
    type: str
    connections: list[(int, int)]


def _expand_values(attr, value, count):
    if isinstance(value, str):
        ds_value = [value] * count
    elif isinstance(value, Sequence):
        assert len(value) == count, f"For {attr}, {len(value)} != (count) {count}"
        ds_value = value
    elif isinstance(value, Iterable):
        ds_value = list(it.islice(value, count))
    else:
        ds_value = [value] * count

    return ds_value


def make_node(filename, name, count, wanted_attributes):
    with h5py.File(filename, "w") as h5:
        dg = h5.create_group(f"/nodes/{name}")

        for attr, value in wanted_attributes.items():
            typ = NODE_TYPES[attr]
            ds_name = ("0/" if typ.prefix else "") + typ.name
            ds_value = _expand_values(attr, value, count)
            dg.create_dataset(name=ds_name, data=ds_value, dtype=typ.type)


def make_edges(filename, edges, wanted_attributes):
    name = f"{edges.src}__{edges.tgt}__{edges.type}"
    src_ids, tgt_ids = zip(*edges.connections)
    count = len(src_ids)
    with h5py.File(filename, "w") as h5:
        dg = h5.create_group(f"/edges/{name}")

        for attr, value in wanted_attributes.items():
            typ = EDGE_TYPES[attr]
            ds_name = ("0/" if typ.prefix else "") + typ.name
            ds_value = _expand_values(attr, value, count)
            dg.create_dataset(name=ds_name, data=ds_value, dtype=typ.type)

        ds = dg.create_dataset("source_node_id", data=np.array(src_ids, dtype=int))
        ds.attrs["node_population"] = edges.src
        ds = dg.create_dataset("target_node_id", data=np.array(tgt_ids, dtype=int))
        ds.attrs["node_population"] = edges.tgt

    libsonata.EdgePopulation.write_indices(
        filename,
        name,
        source_node_count=count,
        target_node_count=count,
    )



def make_ringtest_nodes():
    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:B_BallStick",
        "mtype": "MTYPE",  # neurodamus/io/cell_readers.py:140: SonataError

        # neurodamus/io/cell_readers.py:162: SonataError
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),

        # Note: the morphology isn't used because it's encoded in the hoc file
        "morphology": "NOT_USED",
        }
    make_node(filename="nodes_A.h5", name="RingA", count=3, wanted_attributes=wanted)

    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:B_BallStick",
        "mtype": "MTYPE",  # neurodamus/io/cell_readers.py:140: SonataError

        # neurodamus/io/cell_readers.py:162: SonataError
        "x": it.count(3),
        "y": it.count(4),
        "z": it.count(5),

        # Note: the morphology isn't used because it's encoded in the hoc file
        "morphology": "NOT_USED",
        }
    make_node(filename="nodes_B.h5", name="RingB", count=2, wanted_attributes=wanted)


def make_edges_A():
    edges = Edges("RingA", "RingA", "chemical", [(0, 1), (1, 2), (2, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(11.),
        "decay_time": it.count(12.),
        "delay": it.count(13.),
        "depression_time": it.count(14.),
        "facilitation_time": it.count(15.),
        "u_syn": it.count(16.),

        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,

        "n_rrp_vesicles": 4,
        "syn_type_id": [60, 104, 77],
        }
    make_edges(filename="local_edges_A.h5", edges=edges, wanted_attributes=wanted_attributes)


def make_edges_B():
    edges = Edges("RingB", "RingB", "chemical", [(0, 1), (1, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(21.),
        "decay_time": it.count(22.),
        "delay": it.count(23.),
        "depression_time": it.count(24.),
        "facilitation_time": it.count(25.),
        "u_syn": it.count(26.),

        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,

        "n_rrp_vesicles": 4,
        "syn_type_id": [24, 29],
        }
    make_edges(filename="local_edges_B.h5", edges=edges, wanted_attributes=wanted_attributes)


def make_edges_AB():
    edges = Edges("RingA", "RingB", "chemical", [(0, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": 1.,
        "decay_time": 2.,
        "delay": 3.,
        "depression_time": 4.,
        "facilitation_time": 5.,
        "u_syn": 6.,

        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": 31,
        }
    make_edges(filename="edges_AB.h5", edges=edges, wanted_attributes=wanted_attributes)


make_ringtest_nodes()
make_edges_AB()
make_edges_A()
make_edges_B()
