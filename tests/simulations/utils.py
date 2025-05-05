from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import itertools as it
import h5py
import libsonata
import numpy as np
from typing import Optional

@dataclass
class Edges:
    src: str
    tgt: str
    type: str
    connections: list[(int, int)]
    edge_pop: Optional[str] = None
    synapses: Optional[list[int]] = None


@dataclass
class SonataAttribute:
    name: str
    type: type
    prefix: bool


NODE_TYPES = [
    SonataAttribute("node_type_id", type=int, prefix=False),
    SonataAttribute("model_template", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("model_type", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("mtype", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("etype", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("morphology", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("type", type=int, prefix=True),
    SonataAttribute("start_node", type=int, prefix=True),
    SonataAttribute("end_node", type=int, prefix=True),
    SonataAttribute("section_id", type=int, prefix=True),
    SonataAttribute("segment_id", type=int, prefix=True),
    SonataAttribute("x", type=np.float32, prefix=True),
    SonataAttribute("y", type=np.float32, prefix=True),
    SonataAttribute("z", type=np.float32, prefix=True),
    SonataAttribute("radius", type=np.float32, prefix=True),
    SonataAttribute("start_diameter", type=np.float32, prefix=True),
    SonataAttribute("end_diameter", type=np.float32, prefix=True)
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
    SonataAttribute("afferent_junction_id", type=int, prefix=True),
    SonataAttribute("efferent_junction_id", type=int, prefix=True),
    SonataAttribute("astrocyte_section_id", type=int, prefix=True),
    SonataAttribute("astrocyte_segment_id", type=int, prefix=True),
    SonataAttribute("astrocyte_segment_offset", type=np.float32, prefix=True),
    SonataAttribute("endfoot_compartment_diameter", type=np.float32, prefix=True),
    SonataAttribute("endfoot_compartment_length", type=np.float32, prefix=True),
    SonataAttribute("endfoot_compartment_perimeter", type=np.float32, prefix=True),
    SonataAttribute("endfoot_id", type=int, prefix=True),
    SonataAttribute("vasculature_section_id", type=int, prefix=True),
    SonataAttribute("vasculature_segment_id", type=int, prefix=True),
    SonataAttribute("neuromod_dtc", type=np.float32, prefix=True),
    SonataAttribute("neuromod_strength", type=np.float32, prefix=True)
    

]
EDGE_TYPES = {attr.name: attr for attr in EDGE_TYPES}


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
    
        # virtual population has no attribute
        # but group "0" is required by libsonata function open_population
        if "0" not in dg:
            dg.create_group("0")


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

        if edges.edge_pop and edges.synapses:
            ds = dg.create_dataset("0/synapse_id", data=np.array(edges.synapses, dtype=int))
            ds.attrs["edge_population"] = edges.edge_pop
            ds_value = _expand_values("synapse_population", edges.edge_pop, len(edges.synapses))
            ds = dg.create_dataset("0/synapse_population", data=ds_value, dtype=h5py.string_dtype())

            


    libsonata.EdgePopulation.write_indices(
        filename,
        name,
        source_node_count=max(src_ids) + 1,  # add 1 because IDs are 0-based
        target_node_count=max(tgt_ids) + 1,
    )