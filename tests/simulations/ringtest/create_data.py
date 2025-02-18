#"""
#`lhs > rhs` is an edge from lhs to rhs
#
#Have 2 populations
#
#   nodeA  nodeB  nodeC   <- population
#00 0a  >   0a       0a         ('A', 'B', 0, 0)
#01 1b  <   1b       1b         ('B', 'A', 1, 1)
#02 2a      2a   >   2a         ('B', 'C', 2, 2)
#03 3b      3b   <   3b         ('C', 'B', 3, 3)
#04 4a      4a       4a  <      ('A', 'C', 4, 4)
#05 4a      4a       4a  >      ('C', 'A', 4, 4)
#06 5b  >   5a       5a         ('A', 'B', 5, 5)  \ type b -> a
#07 5b      5a       5a  <      ('A', 'C', 5, 5)  /
#08 0a  >   0a       0a         ('A', 'B', 0, 0)  \ duplicates
#09 1b  <   1b       1b         ('B', 'A', 1, 1)  /
#10 0a  >   2a       2a         ('A', 'B', 0, 2)
# | ||
# | |\ a/b are 'mtypes'
# | \ Node ID
# \ edge id
#
#After keeping only mtypes of 'a' type;
#
#nodeA  nodeB  nodeC
#0a  >   0a       0a                    ('A', 'B', 0, 0)
#        2a   >   2a      Renumbered -> ('B', 'C', 1, 1)
#4a               4a  <                 ('A', 'C', 2, 2)
#0a  >   0a                             ('A', 'B', 0, 0)  duplicate
#0a  >   2a                             ('A', 'B', 0, 1)
#
#Note: Since nodes are being removed, only node IDs 0/2/4 will be kept, and they need to be renumbered
#
#For 'external'; only
# Node  nodaA 5b will be kept; pointing to both nodeB::5a and nodeC::5a, which were renumbered
# 5b  >   5a       5a                   ('external_A', 'B', 0, 5)
# 5b      5a       5a  <  Renumbered -> ('external_A', 'C', 0, 5)
#"""

from collections import namedtuple

import h5py
import libsonata
import numpy as np

type Population = str
type Gid = int

class Edge:
    src: Population
    tgt: Population
    sgid: Gid
    tgid: Gid


def make_edges(edge_file_name, edges):
    def add_data(h5, path, data):
        if path in h5:
            data = np.concatenate((h5[path][:], data))
            del h5[path]

        ds = h5.create_dataset(path, data=data)

        return ds

    pop_names = set()
    with h5py.File(edge_file_name, "w") as h5:
        for e in edges:
            pop_name = f"{e.src}__{e.tgt}"
            pop_names.add(pop_name)
            ds = add_data(h5, f"/edges/{pop_name}/source_node_id", data=np.array(e.sgid, dtype=int))
            ds.attrs["node_population"] = e.src

            ds = add_data(h5, f"/edges/{pop_name}/target_node_id", data=np.array(e.tgid, dtype=int))
            ds.attrs["node_population"] = e.tgt

            add_data(
                h5, f"/edges/{pop_name}/0/delay", data=np.array([0.5] * len(e.tgid), dtype=float)
            )
            add_data(
                h5, f"/edges/{pop_name}/edge_type_id", data=np.array([-1] * len(e.tgid), dtype=int)
            )

    for pop_name in pop_names:
        libsonata.EdgePopulation.write_indices(
            # TODO: should makes sure node count is enough
            edge_file_name,
            pop_name,
            source_node_count=10,
            target_node_count=10,
        )


def make_nodes():
    name = "RingA"
    count = 3
    with h5py.File("nodes_A.h5", "w") as h5:
        h5.create_dataset(f"/nodes/{name}/0/model_template", data=["hoc:B_BallStick"] * count)
        h5.create_dataset(f"/nodes/{name}/0/model_type", data=["biophysical"] * count)
        h5.create_dataset(f"/nodes/{name}/node_type_id", data=[-1] * count)

        # neurodamus/io/cell_readers.py:140: SonataError
        h5.create_dataset(f"/nodes/{name}/0/mtype", data=["MIKE"] * count)

        # neurodamus/io/cell_readers.py:162: SonataError
        h5.create_dataset(f"/nodes/{name}/0/x", data=[0.0] * count, dtype=np.float32)
        h5.create_dataset(f"/nodes/{name}/0/y", data=[0.0] * count, dtype=np.float32)
        h5.create_dataset(f"/nodes/{name}/0/z", data=[0.0] * count, dtype=np.float32)

        # Note: the morphology isn't used because it's encoded in the hoc file
        h5.create_dataset(f"/nodes/{name}/0/morphology", data=["NOT_USED"] * count)

    name = "RingB"
    count = 2
    with h5py.File("nodes_B.h5", "w") as h5:
        h5.create_dataset(f"/nodes/{name}/0/model_template", data=["hoc:B_BallStick"] * count)
        h5.create_dataset(f"/nodes/{name}/0/model_type", data=["biophysical"] * count)
        h5.create_dataset(f"/nodes/{name}/node_type_id", data=[-1] * count)

        # neurodamus/io/cell_readers.py:140: SonataError
        h5.create_dataset(f"/nodes/{name}/0/mtype", data=["MIKE"] * count)

        # neurodamus/io/cell_readers.py:162: SonataError
        h5.create_dataset(f"/nodes/{name}/0/x", data=[0.0] * count, dtype=np.float32)
        h5.create_dataset(f"/nodes/{name}/0/y", data=[0.0] * count, dtype=np.float32)
        h5.create_dataset(f"/nodes/{name}/0/z", data=[0.0] * count, dtype=np.float32)

        # Note: the morphology isn't used because it's encoded in the hoc file
        h5.create_dataset(f"/nodes/{name}/0/morphology", data=["NOT_USED"] * count)

make_nodes()


def make_edges():
    name = "RingA__RingB__chemical"
    count = 1
    path = "edges_AB.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset(f"/edges/{name}/edge_type_id", data=[-1] * count)

        h5.create_dataset(f"/edges/{name}/0/conductance", data=[1.] * count, dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/decay_time", data=[2.] * count, dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/delay", data=[3.] * count, dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/depression_time", data=[4.] * count, dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/facilitation_time", data=[5.] * count, dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/u_syn", data=[6.] * count, dtype=np.float32)

        h5.create_dataset(f"/edges/{name}/0/afferent_section_id", data=[1] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/afferent_section_pos", data=[0.75] * count, dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/afferent_segment_id", data=[1] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/afferent_segment_offset", data=[0] * count, dtype=int)

        h5.create_dataset(f"/edges/{name}/0/n_rrp_vesicles", data=[4] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/syn_type_id", data=[31] * count, dtype=int)

        ds = h5.create_dataset(f"/edges/{name}/source_node_id", data=np.array([0], dtype=int))
        ds.attrs["node_population"] = "RingA"
        ds = h5.create_dataset(f"/edges/{name}/target_node_id", data=np.array([0], dtype=int))
        ds.attrs["node_population"] = "RingB"

    libsonata.EdgePopulation.write_indices(
        path,
        name,
        source_node_count=1,
        target_node_count=1,
    )

make_edges()

def aaa():
    edges = (
        Edge("A", "B", [0, 5], [0, 5]),
        Edge("B", "A", [1], [1]),
        Edge("B", "C", [2], [2]),
        Edge("C", "B", [3], [3]),
        Edge("A", "C", [4, 5], [4, 5]),
        Edge("C", "A", [4], [4]),
        Edge("A", "B", [0], [0]),
        Edge("B", "A", [1], [1]),
        Edge("A", "B", [0], [2]),
    )

    make_edges("edges.h5", edges)

    """
    For the virtual nodes, two separate files, with 2 populations V1, and V2;
    V1 innervates populations A, and B, which V2 innervates C

    nodeV1  A  B                     nodeV1  A  B
    0      >0                        0      >0
    1         >1        --- keep -->
    2         >0                     2         >0
    3      >0                        3      >0

    keep ->

    nodeV2  C
    0      >2
    """

    with h5py.File("virtual_nodes_V1.h5", "w") as h5:
        h5.create_dataset(
            "/nodes/V1/0/model_type",
            data=[
                "virtual",
            ]
            * 4,
        )
        h5.create_dataset(
            "/nodes/V1/node_type_id",
            data=[
                1,
                1,
                1,
                1,
            ],
        )

    edges = (
        Edge("V1", "A", [0, 3], [0, 0]),
        Edge("V1", "B", [1, 2], [1, 0]),
    )
    make_edges("virtual_edges_V1.h5", edges)

    edges = (Edge("V2", "C", [0], [2]),)
    make_edges("virtual_edges_V2.h5", edges)
    with h5py.File("virtual_nodes_V2.h5", "w") as h5:
        h5.create_dataset("/nodes/V2/0/model_type", data=["virtual"] * 1)
        h5.create_dataset("/nodes/V2/node_type_id", data=[1])


#tests/simulations/ringtest/edges_AB.h5
#    /                        Group
#    /edges                   Group
#    /edges/RingA__RingB__chemical Group
#    /edges/RingA__RingB__chemical/0 Group
#    /edges/RingA__RingB__chemical/0/afferent_center_x Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_center_y Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_center_z Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_section_id Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_section_pos Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_section_type Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_segment_id Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_segment_offset Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_surface_x Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_surface_y Dataset {1}
#    /edges/RingA__RingB__chemical/0/afferent_surface_z Dataset {1}
#    /edges/RingA__RingB__chemical/0/conductance Dataset {1}
#    /edges/RingA__RingB__chemical/0/conductance_scale_factor Dataset {1}
#    /edges/RingA__RingB__chemical/0/decay_time Dataset {1}
#    /edges/RingA__RingB__chemical/0/delay Dataset {1}
#    /edges/RingA__RingB__chemical/0/depression_time Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_center_x Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_center_y Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_center_z Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_section_id Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_section_pos Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_section_type Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_segment_id Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_segment_offset Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_surface_x Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_surface_y Dataset {1}
#    /edges/RingA__RingB__chemical/0/efferent_surface_z Dataset {1}
#    /edges/RingA__RingB__chemical/0/facilitation_time Dataset {1}
#    /edges/RingA__RingB__chemical/0/n_rrp_vesicles Dataset {1}
#    /edges/RingA__RingB__chemical/0/spine_length Dataset {1}
#    /edges/RingA__RingB__chemical/0/syn_type_id Dataset {1}
#    /edges/RingA__RingB__chemical/0/u_hill_coefficient Dataset {1}
#    /edges/RingA__RingB__chemical/0/u_syn Dataset {1}
#    /edges/RingA__RingB__chemical/edge_type_id Dataset {1}
#    /edges/RingA__RingB__chemical/indices Group
#    /edges/RingA__RingB__chemical/indices/source_to_target Group
#    /edges/RingA__RingB__chemical/indices/source_to_target/node_id_to_ranges Dataset {3, 2}
#    /edges/RingA__RingB__chemical/indices/source_to_target/range_to_edge_id Dataset {1, 2}
#    /edges/RingA__RingB__chemical/indices/target_to_source Group
#    /edges/RingA__RingB__chemical/indices/target_to_source/node_id_to_ranges Dataset {2, 2}
#    /edges/RingA__RingB__chemical/indices/target_to_source/range_to_edge_id Dataset {1, 2}
#    /edges/RingA__RingB__chemical/source_node_id Dataset {1}
#    /edges/RingA__RingB__chemical/target_node_id Dataset {1}

#tests/simulations/ringtest/local_edges_A.h5
#    /                        Group
#    /edges                   Group
#    /edges/RingA__RingA__chemical Group
#    /edges/RingA__RingA__chemical/0 Group
#    /edges/RingA__RingA__chemical/0/afferent_center_x Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_center_y Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_center_z Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_section_id Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_section_pos Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_section_type Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_segment_id Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_segment_offset Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_surface_x Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_surface_y Dataset {3}
#    /edges/RingA__RingA__chemical/0/afferent_surface_z Dataset {3}
#    /edges/RingA__RingA__chemical/0/conductance Dataset {3}
#    /edges/RingA__RingA__chemical/0/conductance_scale_factor Dataset {3}
#    /edges/RingA__RingA__chemical/0/decay_time Dataset {3}
#    /edges/RingA__RingA__chemical/0/delay Dataset {3}
#    /edges/RingA__RingA__chemical/0/depression_time Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_center_x Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_center_y Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_center_z Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_section_id Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_section_pos Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_section_type Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_segment_id Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_segment_offset Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_surface_x Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_surface_y Dataset {3}
#    /edges/RingA__RingA__chemical/0/efferent_surface_z Dataset {3}
#    /edges/RingA__RingA__chemical/0/facilitation_time Dataset {3}
#    /edges/RingA__RingA__chemical/0/n_rrp_vesicles Dataset {3}
#    /edges/RingA__RingA__chemical/0/spine_length Dataset {3}
#    /edges/RingA__RingA__chemical/0/syn_type_id Dataset {3}
#    /edges/RingA__RingA__chemical/0/u_hill_coefficient Dataset {3}
#    /edges/RingA__RingA__chemical/0/u_syn Dataset {3}
#    /edges/RingA__RingA__chemical/edge_type_id Dataset {3}
#    /edges/RingA__RingA__chemical/indices Group
#    /edges/RingA__RingA__chemical/indices/source_to_target Group
#    /edges/RingA__RingA__chemical/indices/source_to_target/node_id_to_ranges Dataset {3, 2}
#    /edges/RingA__RingA__chemical/indices/source_to_target/range_to_edge_id Dataset {3, 2}
#    /edges/RingA__RingA__chemical/indices/target_to_source Group
#    /edges/RingA__RingA__chemical/indices/target_to_source/node_id_to_ranges Dataset {3, 2}
#    /edges/RingA__RingA__chemical/indices/target_to_source/range_to_edge_id Dataset {3, 2}
#    /edges/RingA__RingA__chemical/source_node_id Dataset {3}
#    /edges/RingA__RingA__chemical/target_node_id Dataset {3}
#
#tests/simulations/ringtest/local_edges_B.h5
#    /                        Group
#    /edges                   Group
#    /edges/RingB__RingB__chemical Group
#    /edges/RingB__RingB__chemical/0 Group
#    /edges/RingB__RingB__chemical/0/afferent_center_x Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_center_y Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_center_z Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_section_id Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_section_pos Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_section_type Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_segment_id Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_segment_offset Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_surface_x Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_surface_y Dataset {2}
#    /edges/RingB__RingB__chemical/0/afferent_surface_z Dataset {2}
#    /edges/RingB__RingB__chemical/0/conductance Dataset {2}
#    /edges/RingB__RingB__chemical/0/conductance_scale_factor Dataset {2}
#    /edges/RingB__RingB__chemical/0/decay_time Dataset {2}
#    /edges/RingB__RingB__chemical/0/delay Dataset {2}
#    /edges/RingB__RingB__chemical/0/depression_time Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_center_x Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_center_y Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_center_z Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_section_id Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_section_pos Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_section_type Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_segment_id Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_segment_offset Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_surface_x Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_surface_y Dataset {2}
#    /edges/RingB__RingB__chemical/0/efferent_surface_z Dataset {2}
#    /edges/RingB__RingB__chemical/0/facilitation_time Dataset {2}
#    /edges/RingB__RingB__chemical/0/n_rrp_vesicles Dataset {2}
#    /edges/RingB__RingB__chemical/0/spine_length Dataset {2}
#    /edges/RingB__RingB__chemical/0/syn_type_id Dataset {2}
#    /edges/RingB__RingB__chemical/0/u_hill_coefficient Dataset {2}
#    /edges/RingB__RingB__chemical/0/u_syn Dataset {2}
#    /edges/RingB__RingB__chemical/edge_type_id Dataset {2}
#    /edges/RingB__RingB__chemical/indices Group
#    /edges/RingB__RingB__chemical/indices/source_to_target Group
#    /edges/RingB__RingB__chemical/indices/source_to_target/node_id_to_ranges Dataset {2, 2}
#    /edges/RingB__RingB__chemical/indices/source_to_target/range_to_edge_id Dataset {2, 2}
#    /edges/RingB__RingB__chemical/indices/target_to_source Group
#    /edges/RingB__RingB__chemical/indices/target_to_source/node_id_to_ranges Dataset {2, 2}
#    /edges/RingB__RingB__chemical/indices/target_to_source/range_to_edge_id Dataset {2, 2}
#    /edges/RingB__RingB__chemical/source_node_id Dataset {2}
#    /edges/RingB__RingB__chemical/target_node_id Dataset {2}
