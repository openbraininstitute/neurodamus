#!/bin/env python
# /// script
# dependencies = ['h5py', 'libsonata', 'numpy']
# ///
# the above allows one to run `uv run create_data.py` without a virtualenv
import sys
from pathlib import Path
import itertools as it
import h5py

# Add path for local imports
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import Edges, make_nodes, make_edges


def make_ringtest_nodes(num_ringA_nodes, num_ringB_nodes):

    # sscx-v7-plasticity
    hocs = ["hoc:cADpyr_L5TPC", "hoc:dSTUT_L2SBC", "hoc:cSTUT_L6NGC",       
        "hoc:cADpyr_L5TPC", "hoc:cADpyr_L5TPC"
        ]
    mtypes = ["L5_TPC:C", "L5_LBC", "L5_NBC", "L5_TPC:B", "L5_TPC:A"]
    etypes = ["cADpyr", "dSTUT", "cSTUT", "cADpyr", "cADpyr"]
    morphologies = [
        "dend-rp100428-3_idD_axon-vd110315_idE_-_Scale_x1.000_y0.975_z1.000_-_Clone_0",
        "mtC050301A_idC_-_Scale_x1.000_y1.025_z1.000_-_Clone_0",
        "rp110711_C3_idA_-_Scale_x1.000_y1.025_z1.000_-_Clone_1",
        "dend-rat_20160303_LH2_E2_cell1_axon-vd111221_idB_-_Scale_x1.000_y0.975_z1.000",
        "dend-vd110504_idA_axon-rat_20160906_E1_LH5_cell2_-_Clone_0",
        ]
    # usecase3
    hocs += ["hoc:cNAC_L23BTC", "hoc:cNAC_L23BTC", "hoc:cADpyr_L2TPC"]
    mtypes += ["L4_PC", "L4_MC", "L4_MC"]
    etypes += ["dSTUT", "dSTUT", "dNAC"]
    morphologies += ["rr110330_C3_idA", "C210401C", "rr110330_C3_idA"]
    # v5_gapjunctions
    hocs += ["hoc:cACint209", "hoc:bIR215"]
    mtypes += ["L5_MC", "L5_MC"]
    etypes += ["cACint", "bIR"]
    morphologies += ["C210301C1_cor", "C290500C-I4"]
    # v5_sonata
    hocs += ["hoc:cNAC187", "hoc:bNAC219", "hoc:cADpyr229",
        "hoc:cACint209", "hoc:cADpyr229"
        ]
    mtypes += ["L1_HAC", "L1_DAC", "L23_PC", "L23_NBC", "L23_PC"]
    etypes += ["cNAC", "bNAC", "cADpyr", "cACint", "cADpyr"]
    morphologies += ["sm080908a4", "sm080902a3-2", 
        "dend-C280199C-P3_axon-C220797A-P1_-_Clone_4", 
        "C040600B3_-_Clone_7","dend-C090905B_axon-C220797A-P1_-_Clone_5"
        ]
    
    wanted = {
        "node_type_id": -1,
        "model_template": it.islice(it.cycle(hocs), num_ringA_nodes),
        "model_type": "biophysical",
        "mtype": it.islice(it.cycle(mtypes), num_ringA_nodes),
        "etype": it.islice(it.cycle(etypes), num_ringA_nodes),
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),
        "morphology": it.islice(it.cycle(morphologies), num_ringA_nodes)
    }
    make_nodes(filename="nodes_A.h5", name="RingA", count=num_ringA_nodes, wanted_attributes=wanted)

    wanted = {
        "node_type_id": -1,
        "model_template": it.islice(it.cycle(hocs), num_ringB_nodes),
        "model_type": "biophysical",
        "mtype": it.islice(it.cycle(mtypes), num_ringB_nodes),
        "etype": it.islice(it.cycle(etypes), num_ringB_nodes),
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),
        "morphology": it.islice(it.cycle(morphologies), num_ringB_nodes)
    }

    make_nodes(filename="nodes_B.h5", name="RingB", count=num_ringB_nodes, wanted_attributes=wanted)


def make_ringtest_edges(num_ringA_nodes, num_ringB_nodes):
    edges = Edges("RingA", "RingA", "chemical", [(i, (i+1)%num_ringA_nodes) for i in range(num_ringA_nodes)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(11.0),
        "decay_time": it.count(12.0),
        "delay": it.count(13.0),
        "depression_time": it.count(14.0),
        "facilitation_time": it.count(15.0),
        "u_syn": it.count(16.0),
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": it.islice(it.cycle([160, 104, 177]),num_ringA_nodes)
    }
    make_edges(filename="local_edges_A.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges("RingB", "RingB", "chemical", [(i, (i+1)%num_ringB_nodes) for i in range(num_ringB_nodes)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(11.0),
        "decay_time": it.count(12.0),
        "delay": it.count(13.0),
        "depression_time": it.count(14.0),
        "facilitation_time": it.count(15.0),
        "u_syn": it.count(16.0),
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": it.islice(it.cycle([160, 104, 177]),num_ringB_nodes)
    }
    make_edges(filename="local_edges_B.h5", edges=edges, wanted_attributes=wanted_attributes)


    edges = Edges("RingA", "RingB", "chemical", [(0, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": 100.0,
        "decay_time": 2.0,
        "delay": 3.0,
        "depression_time": 4.0,
        "facilitation_time": 5.0,
        "u_syn": 6.0,
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": 131,
    }
    make_edges(filename="edges_AB.h5", edges=edges, wanted_attributes=wanted_attributes)


num_ringA_nodes=20
num_ringB_nodes=1000

make_ringtest_nodes(num_ringA_nodes=num_ringA_nodes, num_ringB_nodes=num_ringB_nodes)
make_ringtest_edges(num_ringA_nodes=num_ringA_nodes, num_ringB_nodes=num_ringB_nodes)
