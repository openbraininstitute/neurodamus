#!/bin/env python
# /// script
# dependencies = ['h5py', 'libsonata', 'numpy']
# ///
# the above allows one to run `uv run create_data.py` without a virtualenv

import h5py
import libsonata
import numpy as np


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


def make_edges_AB():
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
        source_node_count=count,
        target_node_count=count,
    )


def make_edges_A():
    name = "RingA__RingA__chemical"
    count = 3
    path = "local_edges_A.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset(f"/edges/{name}/edge_type_id", data=[-1] * count)

        h5.create_dataset(f"/edges/{name}/0/conductance", data=[11., 12., 13.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/decay_time", data=[12., 13., 14.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/delay", data=[13., 14., 15.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/depression_time", data=[14., 15., 16.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/facilitation_time", data=[15., 16., 17.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/u_syn", data=[16., 17., 18,], dtype=np.float32)

        h5.create_dataset(f"/edges/{name}/0/afferent_section_id", data=[1] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/afferent_section_pos", data=[0.75] * count, dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/afferent_segment_id", data=[1] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/afferent_segment_offset", data=[0] * count, dtype=int)

        h5.create_dataset(f"/edges/{name}/0/n_rrp_vesicles", data=[4] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/syn_type_id", data=[60, 104, 77] * count, dtype=int)

        ds = h5.create_dataset(f"/edges/{name}/source_node_id", data=np.array([0, 1, 2], dtype=int))
        ds.attrs["node_population"] = "RingA"
        ds = h5.create_dataset(f"/edges/{name}/target_node_id", data=np.array([1, 2, 0], dtype=int))
        ds.attrs["node_population"] = "RingA"

    libsonata.EdgePopulation.write_indices(
        path,
        name,
        source_node_count=count,
        target_node_count=count,
    )


def make_edges_B():
    name = "RingB__RingB__chemical"
    count = 2
    path = "local_edges_B.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset(f"/edges/{name}/edge_type_id", data=[-1] * count)

        h5.create_dataset(f"/edges/{name}/0/conductance", data=[21., 22.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/decay_time", data=[22., 23.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/delay", data=[23., 24.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/depression_time", data=[24., 25.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/facilitation_time", data=[25., 26.], dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/u_syn", data=[26., 27.], dtype=np.float32)

        h5.create_dataset(f"/edges/{name}/0/afferent_section_id", data=[1] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/afferent_section_pos", data=[0.75] * count, dtype=np.float32)
        h5.create_dataset(f"/edges/{name}/0/afferent_segment_id", data=[1] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/afferent_segment_offset", data=[0] * count, dtype=int)

        h5.create_dataset(f"/edges/{name}/0/n_rrp_vesicles", data=[4] * count, dtype=int)
        h5.create_dataset(f"/edges/{name}/0/syn_type_id", data=[24, 29] * count, dtype=int)

        ds = h5.create_dataset(f"/edges/{name}/source_node_id", data=np.array([0, 1], dtype=int))
        ds.attrs["node_population"] = "RingB"
        ds = h5.create_dataset(f"/edges/{name}/target_node_id", data=np.array([1, 0], dtype=int))
        ds.attrs["node_population"] = "RingB"

    libsonata.EdgePopulation.write_indices(
        path,
        name,
        source_node_count=count,
        target_node_count=count,
    )


make_nodes()
make_edges_AB()
make_edges_A()
make_edges_B()
