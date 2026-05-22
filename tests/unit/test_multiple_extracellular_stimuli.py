from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import find_peaks

from tests.conftest import RINGTEST_DIR
from tests.utils import (
    f_cos,
    make_ramp_envelope,
    read_ascii_report,
    record_compartment_reports,
    write_ascii_reports,
)

from neurodamus import Neurodamus, Node
from neurodamus.core.stimuli import ElectrodeSource
from neurodamus.stimulus_manager import SpatiallyUniformEField

pytestmark = pytest.mark.skip(reason="not implemented yet")

# Reference of various stim vectors without delay
REF_COSINE = np.array(
    [
        -0.0,
        -0.204561,
        -0.156271,
        0.156271,
        0.409122,
        0.505702,
        0.409122,
        0.156271,
        -0.156271,
        -0.409122,
        -0.505702,
        -0.409122,
        -0.156271,
        0.156271,
        0.409122,
        0.337135,
        0.136374,
        0.0,
        0.0,
    ]
)

REF_CONSTANT = np.array(
    [
        -0.0,
        -0.482168,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.64289,
        -0.321445,
        -0.0,
        0.0,
    ]
)

REF_CONSTANT_SMALLCELL = [
    -0.0,
    -8.006978,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -16.013957,
    -10.675971,
    -5.337986,
    -0.0,
    0.0,
]


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_two_stimulus_blocks(create_tmp_simulation_config_file):  # noqa: PLR0914
    """
    Two stimulus blocks, one contains a cosine field and the other contains a constant field,
    they should be summed
    1. check their stimulus managers share the same SpatiallyUniformEField instance (singleton)
    2. check the size of segment_efield_integrators, should be applied to all the segments, n_seg
    3. check ElectrodeSource.fields list contain 2 fields
    4. check potential of 1st segment should be 0 (soma),
       for 4th segment the sum of the cosine fields and constant fields
    5. check an extracellar mechanism is added to each segment
    6. check the long/unused vectors of ElectrodeSource object are cleaned at the end
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    assert len(n._stim_manager._stimulus) == 1
    stimulus = n._stim_manager._stimulus[0]
    assert stimulus == SpatiallyUniformEField._instance
    cell_manager = n.circuits.get_node_manager("RingA")
    cellref = cell_manager.get_cellref(0)
    rec_dend = Nd.Vector()
    rec_dend.record(cellref.dend[0](0.25).extracellular._ref_e)
    rec_soma = Nd.Vector()
    rec_soma.record(cellref.soma[0](0.5).extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    assert list(stimulus.stimList.keys()) == [0]
    es = stimulus.stimList[0]
    assert isinstance(es, ElectrodeSource)
    assert len(es.segment_efield_integrators) == sum(sec.nseg for sec in cellref.all)
    assert len(es.fields) == 2
    d1 = es.fields[0].duration + es.fields[0].ramp_up_time + es.fields[0].ramp_down_time
    d2 = es.fields[1].duration + es.fields[1].ramp_up_time + es.fields[1].ramp_down_time
    dt = Nd.dt
    dend_efi = es.segment_efield_integrators[3]

    t1_vec = np.concatenate([[0], np.arange(dt / 2, d1, dt)])
    ramp1_vec = make_ramp_envelope(
        t1_vec, es.fields[0].ramp_up_time, es.fields[0].ramp_down_time, es.fields[0].duration
    )
    t2_vec = np.concatenate([[0], np.arange(dt / 2, d2, dt)])
    ramp2_vec = make_ramp_envelope(
        t2_vec, es.fields[1].ramp_up_time, es.fields[1].ramp_down_time, es.fields[1].duration
    )
    tot_tvec = np.concatenate([[0], np.arange(dt / 2, Nd.tstop, dt)])
    ref_dend = (
        f_cos(
            t1_vec, es.fields[0].frequency, es.fields[0].phase, dend_efi.get_potential_amplitude(0)
        )
        * ramp1_vec
        + f_cos(
            t2_vec, es.fields[1].frequency, es.fields[1].phase, dend_efi.get_potential_amplitude(1)
        )
        * ramp2_vec
    )
    ref_zeros = np.zeros(len(tot_tvec))
    ref_final = ref_zeros.copy()
    ref_final[: len(ref_dend)] += ref_dend
    npt.assert_allclose(rec_soma, ref_zeros)
    npt.assert_allclose(rec_dend, ref_final)

    assert all(sec.has_membrane("extracellular") for sec in cellref.all)
    assert es.segment_displacements is None


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 30},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 4.5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_two_stimulus_blocks_delay(create_tmp_simulation_config_file, capsys):
    """
    1. Check the combination of two stimulus blocks, one with delay.
    2. The original delay 4.5 is not divisible by dt 1.0, rounded up to the next time point.
    3. Record one segment extracellular._ref_e and run simulation, check the values
    """
    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    captured = capsys.readouterr()
    warning = (
        "[WARNING] SpatiallyUniformEField delay 4.5 is not divisible by dt 1.0, "
        "rounded up to the next time point 5.0"
    )
    assert warning in captured.out
    delay = stimulus.delay
    npt.assert_approx_equal(delay, 5)
    ref_timevec = [0, *np.arange(delay, delay + duration + dt + 0.1, dt)]
    ref_stimvec = np.zeros(len(ref_timevec))
    es = stimulus.stimList[0]
    soma_stim_vec = es.segment_potentials[0]
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(soma_stim_vec, ref_stimvec)
    dend_stim_vec = es.segment_potentials[3]
    npt.assert_allclose(dend_stim_vec, np.append(0, REF_COSINE + REF_CONSTANT), rtol=1e-5)

    # record segment extracellular_e, check result after simulation run
    cell_manager = n.circuits.get_node_manager("RingA")
    dend_seg = cell_manager.get_cellref(0).dend[0](0.25)
    rec_seg_e = Nd.h.Vector()
    rec_seg_e.record(dend_seg.extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()
    assert len(rec_seg_e) == 31
    # check extracellular_e has been assigned to the correct values
    npt.assert_allclose(
        rec_seg_e[6:25],
        REF_COSINE + REF_CONSTANT,
        rtol=1e-5,
    )
    # check extracellular_e is back to 0 before and after injection
    npt.assert_allclose(rec_seg_e[0:5], 0)
    npt.assert_allclose(rec_seg_e[26:], 0)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_3": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 200, "Ey": -100, "Ez": 100, "frequency": 0.001},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_three_stimulus_blocks_delay(create_tmp_simulation_config_file):
    """
    Check three fields in the stimlus, cosine + constant + cosine with small freq(almost constant)
    """

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    delay = stimulus.delay
    npt.assert_approx_equal(delay, 5)
    ref_timevec = [0, *np.arange(delay, delay + duration + dt + 0.1, dt)]
    ref_stimvec = np.zeros(len(ref_timevec))
    es = stimulus.stimList[0]
    soma_stim_vec = es.segment_potentials[0]
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(soma_stim_vec, ref_stimvec)
    dend_stim_vec = es.segment_potentials[3]
    npt.assert_allclose(
        dend_stim_vec,
        np.append(0, REF_COSINE + REF_CONSTANT + 2 * REF_CONSTANT),
        rtol=1e-6,
    )
    n.clear_model()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_two_blocks_nodeset_overlap(create_tmp_simulation_config_file):
    """
    Check two stimulus blocks where the node_sets overlap
    Cell 0 contains the sum of the two stimuli
    cell 1,2 contain the 2nd stimluli
    """

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    delay = stimulus.delay
    npt.assert_approx_equal(delay, 5)
    ref_timevec = [0, *np.arange(delay, delay + duration + dt + 0.1, dt)]
    # cell 0
    es0 = stimulus.stimList[0]
    dend_stim_vec = es0.segment_potentials[3]
    npt.assert_allclose(es0.time_vec, ref_timevec)
    npt.assert_allclose(dend_stim_vec, np.append(0, REF_COSINE + REF_CONSTANT), rtol=1e-5)

    # cell 1
    es1 = stimulus.stimList[1]
    dend_stim_vec = es1.segment_potentials[3]
    npt.assert_allclose(es1.time_vec, ref_timevec)
    npt.assert_allclose(dend_stim_vec, np.append(0, REF_CONSTANT_SMALLCELL), rtol=1e-5)
    n.clear_model()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 5,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 7,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_two_blocks_time_overlap(create_tmp_simulation_config_file):
    """
    Check 2 stimulus blocks with different time window
    block 1 : [5,17], block 2: [0,7]
    """

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    delay = stimulus.delay
    npt.assert_approx_equal(delay, 0)  # delay of the last stimulus block
    ref_timevec = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
    ]
    ref_stimvec = [
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -0.964335,
        -1.168896,
        -1.120606,
        0.156271,
        0.409122,
        0.505702,
        0.409122,
        0.156271,
        -0.156271,
        -0.409122,
        -0.337135,
        -0.136374,
        -0.0,
        0.0,
    ]
    es = stimulus.stimList[0]
    dend_stim_vec = es.segment_potentials[3]
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(dend_stim_vec, ref_stimvec, rtol=1e-5)
    n.clear_model()


@pytest.mark.parametrize(
    ("create_tmp_simulation_config_file", "ref_peak"),
    [
        (
            {
                "simconfig_fixture": "ringtest_baseconfig",
                "extra_config": {
                    "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                    "target_simulator": "NEURON",
                    "inputs": {
                        "Stimulus": {
                            "module": "pulse",
                            "input_type": "current_clamp",
                            "delay": 5,
                            "duration": 50,
                            "node_set": "RingA",
                            "represents_physical_electrode": True,
                            "amp_start": 10,
                            "width": 1,
                            "frequency": 50,
                        },
                        "ex_efields_1": {
                            "input_type": "extracellular_stimulation",
                            "module": "spatially_uniform_e_field",
                            "delay": 0,
                            "duration": 50,
                            "node_set": "RingA_Cell0",
                            "fields": [
                                {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 10},
                            ],
                            "ramp_up_time": 1.0,
                            "ramp_down_time": 2.0,
                        },
                        "ex_efields_2": {
                            "input_type": "extracellular_stimulation",
                            "module": "spatially_uniform_e_field",
                            "delay": 0,
                            "duration": 50,
                            "node_set": "RingA_Cell0",
                            "fields": [
                                {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                            ],
                            "ramp_up_time": 1.0,
                            "ramp_down_time": 2.0,
                        },
                    },
                    "reports": {
                        "compartment_v": {
                            "type": "compartment",
                            "cells": "Mosaic",
                            "variable_name": "v",
                            "sections": "all",
                            "dt": 1,
                            "start_time": 0.0,
                            "end_time": 20.0,
                        }
                    },
                },
            },
            [
                6,
                21,
                27,
                42,
                48,
                63,
                69,
                90,
                111,
                132,
                153,
                174,
                195,
                216,
                237,
                258,
                279,
                300,
                321,
                342,
                363,
                384,
                405,
                426,
                447,
                468,
            ],
        ),
    ],
    indirect=["create_tmp_simulation_config_file"],
)
def test_neuron_report_with_efields(create_tmp_simulation_config_file, ref_peak):
    """NEURON integration tests, check the compartment ASCII report"""
    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    n = Neurodamus(create_tmp_simulation_config_file)
    ascii_recorders = record_compartment_reports(n._target_manager)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    # Write ASCII reports
    write_ascii_reports(ascii_recorders, n._run_conf.output_root)

    # Read ASCII reports
    report = Path(n._run_conf.output_root) / ("compartment_v.txt")
    assert report.exists()
    data = read_ascii_report(report)
    cell0_voltage_vec = [vec[3] for vec in data if vec[0] == 0]
    assert len(cell0_voltage_vec) == 21 * 23  # 21 time steps * 23 compartments
    peaks_pos = find_peaks(cell0_voltage_vec, prominence=1)[0]
    npt.assert_allclose(peaks_pos, ref_peak)
    n.clear_model()  # clear up the reporting vector, required for the next run.
