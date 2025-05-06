#!/bin/env python
# objective of this script: create a functioning CI testing
# ngv circuit. We do not necessarily want real data, just
# something that qualitatively runs
import sys
import itertools as it
from pathlib import Path

# Add path for local imports
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import Edges, make_nodes, make_edges


def make_ngv_nodes():
    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:TestCell",
        "model_type": "biophysical",
        "mtype": "MTYPE",
        "etype": "ETYPE",
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),
        "morphology": "cell_small",
    }
    make_nodes(filename="nodes.h5", name="RingA", count=3, wanted_attributes=wanted)

    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:astrocyte",
        "model_type": "astrocyte",
        "mtype": "MTYPE",
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),
        "radius": 4.5,
        "morphology": "glia",
    }
    make_nodes(filename="astrocytes.h5", name="AstrocyteA", count=2, wanted_attributes=wanted)

    num_nodes = 6
    wanted = {
        "node_type_id": -1,
        "model_type": "vasculature",
        "start_node": list(range(0, num_nodes)),
        "end_node": list(range(1, num_nodes+1)),
        "start_diameter": [round(1.1 + 0.1 * i, 1) for i in range(num_nodes)],
        "end_diameter": [round(1.2 + 0.1 * i, 1) for i in range(num_nodes)],
        "section_id": [i // 3 for i in range(num_nodes)],
        "segment_id": [i % 3 for i in range(num_nodes)],
        "type" : 0
    }
    make_nodes(filename="vasculature.h5", name="VasculatureA", count=num_nodes, wanted_attributes=wanted)


def make_ngv_edges():
    edges = Edges("RingA", "RingA", "chemical", [(0, 1), (1, 2), (2, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(100.0),
        "decay_time": it.count(2.0),
        "delay": it.count(3.0),
        "depression_time": it.count(4.0),
        "facilitation_time": it.count(5.0),
        "u_syn": it.count(6.0),
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": [131, 104, 77],
    }
    make_edges(filename="edges.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges(
        "AstrocyteA", "RingA", "synapse_astrocyte", [(0, 0), (1, 1)],
        "RingA__RingA__chemical", [0, 1])
    wanted_attributes = {
        "edge_type_id": -1,
        "astrocyte_section_id" : [4, 27],
        "astrocyte_segment_id" : [0, 1],
        "astrocyte_segment_offset" : [3., 1.]
    }
    make_edges(filename="neuroglia.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges("vasculature", "AstrocyteA", "endfoot", [(2, 0), (4, 1)])
    wanted_attributes = {
        "edge_type_id": -1,
        "astrocyte_section_id" : [5, 28],
        "endfoot_compartment_length": 10.,
        "endfoot_compartment_diameter": 5.,
        "endfoot_compartment_perimeter": 7.
    }
    make_edges(filename="gliovascular.h5", edges=edges, wanted_attributes=wanted_attributes)


make_ngv_nodes()
make_ngv_edges()
