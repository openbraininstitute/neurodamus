#!/bin/env python
# /// script
# dependencies = ['h5py', 'libsonata', 'numpy']
# ///
# the above allows one to run `uv run create_data.py` without a virtualenv
import itertools as it
import sys
from pathlib import Path

import h5py
import numpy as np

# Add path for local imports
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import Edges, make_nodes, make_edges


def _write_single_compartment_report(path, t_vec, values, population, node_id):
    times = np.asarray(t_vec, dtype=np.float32)
    data = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    dt = float(times[1] - times[0])
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as h5f:
        h5f.create_group("report")
        gpop = h5f.create_group(f"/report/{population}")
        ddata = gpop.create_dataset("data", data=data, dtype=np.float32)
        ddata.attrs.create("units", data="nA", dtype=string_dtype)

        gmapping = h5f.create_group(f"/report/{population}/mapping")
        dnodes = gmapping.create_dataset("node_ids", data=[node_id], dtype=np.uint64)
        dnodes.attrs.create("sorted", data=True, dtype=np.uint8)
        gmapping.create_dataset("index_pointers", data=[0, 1], dtype=np.uint64)
        gmapping.create_dataset("element_ids", data=[0], dtype=np.uint32)
        dtimes = gmapping.create_dataset(
            "time", data=[times[0], times[-1], dt], dtype=np.double
        )
        dtimes.attrs.create("units", data="ms", dtype=string_dtype)


def create_current_stimulus():
    sin_stim = {
        "Pattern": "Sinusoidal",
        "Mode": "Current",
        "AmpStart": 1.0,
        "Frequency": 10.0,
        "Duration": 20.0,
        "Delay": 5.0,
        "Dt": 0.1,
        "RepresentsPhysicalElectrode": True,
    }

    nd = Neurodamus(create_tmp_simulation_config_file)
    src_manager = smng.StimulusManager(nd._target_manager)
    src_manager.interpret(target_onecell, sin_stim)
    src_stimulus = src_manager._stimulus[0]
    src_source = src_stimulus.stimList[0]
    src_clamp = next(iter(src_source._clamps)).clamp

    src_t = Nd.Vector()
    src_i = Nd.Vector()
    src_t.record(Nd._ref_t)
    src_i.record(src_clamp._ref_amp)

    Nd.finitialize()
    nd.run()

    report_path = tmp_path / "sinusoidal_replay.h5"
    _write_single_compartment_report(report_path, src_t, src_i, population="RingA", node_id=0)


def make_lfp_weights():
    filename = "lfp_file.h5"
    with h5py.File(filename, "w") as h5:
        def write_pop(population, node_ids, offsets, scaling_factors):
            dg = h5.create_group(population)
            dg.create_dataset("node_ids", data=node_ids)
            dg.create_dataset("offsets", data=offsets)

            dg = h5.create_group("electrodes/" + population)
            dg.create_dataset("scaling_factors", dtype='f8', data=scaling_factors)

        node_ids = [0, 1, 2]
        offsets = [0, 5, 10, 15]
        scaling_factors = [
            [0.011, 0.012],
            [0.021, 0.022],
            [0.031, 0.032],
            [0.041, 0.042],
            [0.051, 0.052],
            [0.061, 0.062],
            [0.071, 0.072],
            [0.081, 0.082],
            [0.091, 0.092],
            [0.101, 0.102],
            [0.111, 0.112],
            [0.121, 0.122],
            [0.131, 0.132],
            [0.141, 0.142],
            [0.151, 0.152]
            ]
        write_pop("RingA", node_ids, offsets, scaling_factors)

        node_ids = [0, 1]
        offsets = [0, 5, 10]
        scaling_factors = [
            [0.014, 0.015, 0.016],
            [0.024, 0.025, 0.026],
            [0.034, 0.035, 0.036],
            [0.044, 0.045, 0.046],
            [0.054, 0.055, 0.056],
            [0.064, 0.065, 0.066],
            [0.074, 0.075, 0.076],
            [0.084, 0.085, 0.086],
            [0.094, 0.095, 0.096],
            [0.104, 0.105, 0.106]
            ]
        write_pop("RingB", node_ids, offsets, scaling_factors)


def make_ringtest_nodes():
    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:TestCell",
        "model_type": "biophysical",
        "mtype": ["MTYPE0", "MTYPE1", "MTYPE2"],
        "etype": ["ETYPE0", "ETYPE1", "ETYPE2"],
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),
        "morphology": "cell_small",
    }
    make_nodes(filename="nodes_A.h5", name="RingA", count=3, wanted_attributes=wanted)

    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:TestCell",
        "model_type": "biophysical",
        "mtype": ["MTYPE0", "MTYPE1", "MTYPE2"],
        "etype": ["ETYPE0", "ETYPE1", "ETYPE2"],
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),
        "morphology": ["cell_big", "cell_small", "cell_small"],
        "orientation_w": -0.233117,
        "orientation_x": 0.136089,
        "orientation_y": -0.825741,
        "orientation_z": 0.556314,
    }
    make_nodes(filename="nodes_A_bigcell.h5", name="RingA", count=3, wanted_attributes=wanted)

    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:TestCell",
        "mtype": ["MTYPE0", "MTYPE1"],
        "etype": ["ETYPE1", "ETYPE1"],
        "x": it.count(3),
        "y": it.count(4),
        "z": it.count(5),
        "morphology": "cell_small",
    }
    make_nodes(filename="nodes_B.h5", name="RingB", count=2, wanted_attributes=wanted)

    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:TestCell",
        "mtype": "MTYPE0",
        "etype": "ETYPE1",
        "x": it.count(3),
        "y": it.count(4),
        "z": it.count(5),
        "morphology": "cell_small",
    }
    make_nodes(filename="nodes_C.h5", name="RingC", count=3, wanted_attributes=wanted)


def make_ringtest_edges():
    edges = Edges("RingA", "RingA", "chemical", [(0, 1), (1, 2), (2, 0)])
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
        "syn_type_id": [60, 104, 77],
    }
    make_edges(filename="local_edges_A.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges("RingB", "RingB", "chemical", [(0, 1), (1, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(21.0),
        "decay_time": it.count(22.0),
        "delay": it.count(23.0),
        "depression_time": it.count(24.0),
        "facilitation_time": it.count(25.0),
        "u_syn": it.count(26.0),
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": [24, 29],
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

    edges = Edges("RingC", "RingC", "electrical", [(0, 2), (2, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": 100.0,
        "afferent_section_id": 1,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "efferent_junction_id": [0, 2],
        "afferent_junction_id": [2, 0]
    }
    make_edges(
        filename="local_edges_C_electrical.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges("RingC", "RingC", "chemical", [(0, 1), (1, 2), (2, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(31.0),
        "decay_time": it.count(32.0),
        "delay": it.count(33.0),
        "depression_time": it.count(34.0),
        "facilitation_time": it.count(35.0),
        "u_syn": it.count(36.0),
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": [60, 104, 77],
    }
    make_edges(
        filename="local_edges_C.h5", edges=edges, wanted_attributes=wanted_attributes)

# For neuromodulation test: Create another A->B edge with different afferent_section_pos
# w.r.t B->B edge
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
        "afferent_section_pos": 0.25,
        "afferent_segment_id": 1,
        "afferent_segment_offset": 0,
        "n_rrp_vesicles": 4,
        "syn_type_id": 131,
    }
    make_edges(
        filename="neuromodulation/edges_AB.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges("virtual_neurons", "RingB", "neuromodulatory", [(0, 0), (1, 0), (1, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "afferent_section_id": 1,
        "afferent_section_pos": [0.22, 0.78, 0.81],
        "afferent_segment_id": 1,
        "delay": it.count(44.0),
        "neuromod_dtc": [50, 75, 75],
        "neuromod_strength": [0.5, 0.2, 0.2]
    }
    make_edges(
        filename="neuromodulation/projections.h5", edges=edges, wanted_attributes=wanted_attributes)

    wanted = {
        "node_type_id": -1,
    }
    make_nodes(
        filename="neuromodulation/virtual_neurons.h5", name="virtual_neurons",
        count=2, wanted_attributes=wanted)


make_ringtest_nodes()
make_ringtest_edges()
make_lfp_weights()
