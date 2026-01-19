from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import find_peaks

from tests.conftest import RINGTEST_DIR
from tests.utils import read_ascii_report, record_compartment_reports, write_ascii_reports

from neurodamus import Neurodamus, Node
from neurodamus.core.configuration import ConfigurationError
from neurodamus.core.stimuli import ElectrodeSource
from neurodamus.stimulus_manager import SpatiallyUniformEField


def test_apply_ramp():
    """Test the function apply_ramp
    case 1: ramp_up/down_time > dt, is a multiple of dt
    case 2: ramp_up/down_time > dt, not  a multiple of dt
    case 3: ramp_up/down_time < dt
    """
    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    # case 1
    dt = 0.5
    ref_up_time = 2  # 4 time steps from 0
    ref_down_time = 1.5  # 2 time steps to 0

    stimulus = ElectrodeSource(0, 0, 100, [], ref_up_time, ref_down_time, dt)
    stim_vec = Nd.Vector(range(1, 11))
    assert np.isclose(stimulus.ramp_up_time, ref_up_time)
    assert np.isclose(stimulus.ramp_down_time, ref_down_time)
    assert np.isclose(stimulus.dt, dt)
    stimulus.apply_ramp(stim_vec, stimulus.dt)
    assert np.allclose(stim_vec.as_numpy(), [0, 2 / 3, 2, 4, 5, 6, 7, 8, 4.5, 0])

    # case 2
    dt = 0.5
    ref_up_time = 2.4  # 4 time steps from 0
    ref_down_time = 1.7  # 3 time steps to 0

    stimulus = ElectrodeSource(0, 0, 100, [], ref_up_time, ref_down_time, dt)
    stim_vec = Nd.Vector(range(1, 11))
    assert np.isclose(stimulus.ramp_up_time, ref_up_time)
    assert np.isclose(stimulus.ramp_down_time, ref_down_time)
    assert np.isclose(stimulus.dt, dt)
    stimulus.apply_ramp(stim_vec, stimulus.dt)
    assert np.allclose(stim_vec.as_numpy(), [0, 2 / 3, 2, 4, 5, 6, 7, 8, 4.5, 0])

    # case 3
    dt = 0.5
    ref_up_time = 0.3
    ref_down_time = 0.4

    stimulus = ElectrodeSource(0, 0, 100, [], ref_up_time, ref_down_time, dt)
    stim_vec = Nd.Vector(range(1, 11))
    assert np.isclose(stimulus.ramp_up_time, ref_up_time)
    assert np.isclose(stimulus.ramp_down_time, ref_down_time)
    assert np.isclose(stimulus.dt, dt)
    stimulus.apply_ramp(stim_vec, stimulus.dt)
    assert np.allclose(stim_vec.as_numpy(), list(range(1, 11)))


def test_interpolate_axon_coordinates():
    """Test interpolate axon's coordinates along y-axis from the soma position"""
    from neurodamus.core import NeuronWrapper as Nd

    soma_position = [-0.60355, 4.2, 2.0]
    section = Nd.h.Section(name="TestCell[0].axon[0]")
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=0)
    npt.assert_allclose(position, [-0.60355, 4.2, 2.0])
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=0.5)
    npt.assert_allclose(position, [-0.60355, 4.2 - 30 * 0.5, 2.0])
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=1)
    npt.assert_allclose(position, [-0.60355, 4.2 - 30, 2.0])
    section = Nd.h.Section(name="TestCell[0].axon[1]")
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=0)
    npt.assert_allclose(position, [-0.60355, 4.2 - 30, 2.0])
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=0.8)
    npt.assert_allclose(position, [-0.60355, 4.2 - 30 - 30 * 0.8, 2.0])
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=1)
    npt.assert_allclose(position, [-0.60355, 4.2 - 60, 2.0])


def test_interpolate_myelin_coordinates():
    """Test interpolate myelin's coordinates along y-axis from the soma position"""
    from neurodamus.core import NeuronWrapper as Nd

    soma_position = [-0.60355, 4.2, 2.0]
    section = Nd.h.Section(name="TestCell[0].myelin[0]")
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=0)
    npt.assert_allclose(position, [-0.60355, 4.2 - 60, 2.0])
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=0.4)
    npt.assert_allclose(position, [-0.60355, 4.2 - 60 - 1000 * 0.4, 2.0])
    position = SpatiallyUniformEField.get_segment_position([], soma_position, section, x=1)
    npt.assert_allclose(position, [-0.60355, 4.2 - 1060, 2.0])


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "one_efield": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [{"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100}],
                        "ramp_up_time": 0,
                        "ramp_down_time": 0,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_one_field_noramp(create_tmp_simulation_config_file):
    """
    One cosinusoid field without ramp
    1. check the size of segment_potentials, should be applied to all the segments, n_seg
    2. check time_vec of stimulus, no ramp_up_time and ramp_down_time
    3. check potentials of 1st segment should be 0 (soma), and a cosine wave for 4th segment
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    assert list(stimulus.stimList.keys()) == [0]  # one object per cell
    cell = cellref = n.circuits.get_node_manager("RingA").get_cell(0)
    cellref = cell.CellRef
    es = stimulus.stimList[0]
    assert isinstance(es, ElectrodeSource)
    total_segments = sum(sec.nseg for sec in cellref.all)
    assert len(es.segment_potentials) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(es.segment_potentials[0], ref_stimvec)
    ref_stimvec = [
        -0.505702,
        -0.409122,
        -0.156271,
        0.156271,
        0.409122,
        0.505702,
        0.409122,
        0.156271,
        -0.156271,
        -0.409122,
        -0.505702,
        0.0,
    ]
    npt.assert_allclose(es.segment_potentials[3], ref_stimvec, rtol=1e-5)

    n.clear_model()


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


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "one_efield": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [{"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100}],
                        "ramp_up_time": 3,
                        "ramp_down_time": 4,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_one_field_withramp(create_tmp_simulation_config_file):
    """
    A cosinusoid field with ramp up and down
    1. check the size of segment_potentials, should be applied to all the segments, n_seg
    2. check time_vec of the stimlus, should include ramp_up_time and ramp_down_time
    3. check potentials of 1st segment should be 0 (soma),
    and a cosine wave with 3 ramp up steps and 4 ramp down steps for 4th segment
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    es = stimulus.stimList[0]
    assert isinstance(es, ElectrodeSource)
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(es.segment_potentials) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(es.segment_potentials[0], ref_stimvec)
    npt.assert_allclose(es.segment_potentials[3], REF_COSINE, rtol=1e-5)
    n.clear_model()


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


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "one_efield": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [{"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0}],
                        "ramp_up_time": 3,
                        "ramp_down_time": 4,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_one_constant_field(create_tmp_simulation_config_file):
    """
    A constant field when frequency = 0
    1. check the size of segment_potentials, should be applied to all the segments, n_seg
    2. check time_vec of stimulus, should include ramp_up_time and ramp_down_time
    3. check potential of 1st segment should be 0 (soma),
    and a constant vec for 4th segment including ramp up and down
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    es = stimulus.stimList[0]
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(es.segment_potentials) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(es.segment_potentials[0], ref_stimvec)
    npt.assert_allclose(es.segment_potentials[3], REF_CONSTANT, rtol=1e-6)
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
                    "ex_efields": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_two_fields(create_tmp_simulation_config_file):
    """
    Two fields that should be summed together cosine + constant fields
    1. check the size of segment_potentials, should be applied to all the segments, n_seg
    2. check time_vec of stimulus, should include ramp_up_time and ramp_down_time
    3. check potential of 1st segment should be 0 (soma),
       for 4th segment the sum of the cosine fields and constant fields
    4. check an extracellar mechanism is added to each segment
    5. check the long/unused vectors of ElectrodeSource object are cleaned at the end
    """

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    es = stimulus.stimList[0]
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(es.segment_potentials) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(es.segment_potentials[0], ref_stimvec)
    npt.assert_allclose(es.segment_potentials[3], REF_COSINE + REF_CONSTANT, rtol=1e-5)

    assert all(sec.has_membrane("extracellular") for sec in cell.all)

    assert es.efields is None
    assert es.segment_displacements is None

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
                    "ex_efields": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_two_fields_delay(create_tmp_simulation_config_file):
    """
    Check the delay is applied correctly into the stimulus segment_potentials and time_vec
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
    ref_timevec = [0, *np.arange(delay, delay + duration + 1, dt), delay + duration]
    ref_stimvec = np.zeros(len(ref_timevec))
    es = stimulus.stimList[0]
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(es.segment_potentials[0], ref_stimvec)
    npt.assert_allclose(
        es.segment_potentials[3], np.append(0, REF_COSINE + REF_CONSTANT), rtol=1e-5
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
                    "ex_efields": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                            {"Ex": 200, "Ey": -100, "Ez": 100, "frequency": 0.001},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_three_fields_delay(create_tmp_simulation_config_file):
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
    ref_timevec = [0, *np.arange(delay, delay + duration + 1, dt), delay + duration]
    ref_stimvec = np.zeros(len(ref_timevec))
    es = stimulus.stimList[0]
    seg_stimuli = list(es.segment_potentials)
    soma_stim_vec = seg_stimuli[0]
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(soma_stim_vec, ref_stimvec)
    dend_stim_vec = seg_stimuli[3]
    npt.assert_allclose(
        dend_stim_vec,
        np.append(0, REF_COSINE + REF_CONSTANT + 2 * REF_CONSTANT),
        rtol=1e-6,
    )
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
                        }
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
                27,
                48,
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
                        "ex_efields": {
                            "input_type": "extracellular_stimulation",
                            "module": "spatially_uniform_e_field",
                            "delay": 0,
                            "duration": 50,
                            "node_set": "RingA_Cell0",
                            "fields": [
                                {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 10},
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
    """2 NEURON integration tests, with and without the electric field stimulus.
    Check the compartment ASCII reports without different reference peak position"""
    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    n = Neurodamus(create_tmp_simulation_config_file)
    ascii_recorders = record_compartment_reports(n._target_manager)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    # Write ASCII reports
    write_ascii_reports(ascii_recorders, n._run_conf["OutputRoot"])

    # Read ASCII reports
    report = Path(n._run_conf["OutputRoot"]) / ("compartment_v.txt")
    assert report.exists()
    data = read_ascii_report(report)
    cell0_voltage_vec = [vec[3] for vec in data if vec[0] == 0]
    assert len(cell0_voltage_vec) == 21 * 23  # 21 time steps * 23 compartments
    peaks_pos = find_peaks(cell0_voltage_vec, prominence=1)[0]
    npt.assert_allclose(peaks_pos, ref_peak)
    n.clear_model()  # clear up the reporting vector, required for the next run.


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "CORENEURON",
                "inputs": {
                    "ex_efields": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "Mosaic",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
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
def test_coreneuron_exception(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    with pytest.raises(
        ConfigurationError,
        match="CoreNEURON cannot simulate a model that contains the extracellular mechanism",
    ):
        Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
