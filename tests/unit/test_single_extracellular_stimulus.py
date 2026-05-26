from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import find_peaks

from tests.conftest import RINGTEST_DIR
from tests.utils import read_ascii_report, record_compartment_reports, write_ascii_reports

from neurodamus import Neurodamus
from neurodamus.core.configuration import ConfigurationError
from neurodamus.core.stimuli import ElectrodeSource
from neurodamus.stimulus_manager import SpatiallyUniformEField


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
                "run": {"dt": 1, "tstop": 10},
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
def test_one_field_noramp(create_tmp_simulation_config_file):  # noqa: PLR0914
    """
    One cosinusoid field without ramp
    1. check the size of segment_potentials, should be applied to all the segments, n_seg
    2. check the reference potentials against the cosine function
    3. check potentials of 1st segment should be 0 (soma), and a cosine wave for 4th segment
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    assert list(stimulus.stimList.keys()) == [0]  # one object per cell
    cell = cellref = n.circuits.get_node_manager("RingA").get_cell(0)
    cellref = cell.CellRef
    es = stimulus.stimList[0]
    assert isinstance(es, ElectrodeSource)
    total_segments = sum(sec.nseg for sec in cellref.all)
    assert len(es.segment_efield_integrators) == total_segments

    soma_seg = cellref.soma[0](0.5)
    dend_seg = cellref.dend[0](0.25)
    rec_dend = Nd.Vector()
    rec_dend.record(dend_seg.extracellular._ref_e)
    rec_soma = Nd.Vector()
    rec_soma.record(soma_seg.extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    dt = es.dt
    duration = es.duration
    efi = es.segment_efield_integrators[3]
    max_potential = 1e3 * (
        efi.displacementX * es.fields[0]["Ex"]
        + efi.displacementY * es.fields[0]["Ey"]
        + efi.displacementZ * es.fields[0]["Ez"]
    )  # from mV to V

    def f_cos(t):
        return max_potential * np.cos(
            2 * np.pi * es.fields[0]["Frequency"] / 1000 * t + es.fields[0]["Phase"]
        )

    # original reference with vec.play with time points at every dt
    ref_stimvec_vecplay = [
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
    t_vecplay = np.arange(0, duration + 1, dt)
    # current reference with efield_integrator.mod at very t+dt/2 w.r.t BEFORE_BREAKPOINT
    ref_stimvec_mod = [
        0,
        -0.4809515,
        -0.2972444,
        0,
        0.2972444,
        0.4809515,
        0.4809515,
        0.2972444,
        0,
        -0.2972444,
        -0.4809515,
    ]
    t_beforebreakpoint = np.arange(dt / 2, duration, dt)

    # check the references against the cosine function
    npt.assert_allclose(f_cos(t_beforebreakpoint), ref_stimvec_mod[1:], atol=1e-9)
    npt.assert_allclose(f_cos(t_vecplay), ref_stimvec_vecplay[:-1], atol=1e-6)

    # check the actually segment.extracellular._ref_e against the current reference
    npt.assert_allclose(rec_dend, ref_stimvec_mod, atol=1e-9)
    npt.assert_allclose(rec_soma, np.zeros(len(rec_dend)))


REF_COSINE = np.array(
    [
        0,
        -0.08015859,
        -0.1486222,
        0,
        0.2972444,
        0.4809515,
        0.4809515,
        0.2972444,
        0,
        -0.2972444,
        -0.4809515,
        -0.4809515,
        -0.2972444,
        0,
        0.26008885,
        0.3005947,
        0.18035683,
        0.03715555,
    ]
)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 17},
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
def test_one_field_withramp(create_tmp_simulation_config_file):  # noqa: PLR0914
    """
    A cosinusoid field with ramp up and down
    1. check the size of segment_potentials, should be applied to all the segments, n_seg
    2. check the reference potentials against the cosine function with ramping
    3. check potentials of 1st segment should be 0 (soma),
    and a cosine wave with 3 ramp up steps and 4 ramp down steps for 4th segment
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    es = stimulus.stimList[0]
    assert isinstance(es, ElectrodeSource)
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(es.segment_efield_integrators) == total_segments

    soma_seg = cell.soma[0](0.5)
    dend_seg = cell.dend[0](0.25)
    rec_dend = Nd.Vector()
    rec_dend.record(dend_seg.extracellular._ref_e)
    rec_soma = Nd.Vector()
    rec_soma.record(soma_seg.extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()
    npt.assert_allclose(rec_dend, REF_COSINE, atol=1e-9)
    npt.assert_allclose(rec_soma, np.zeros(len(rec_dend)))

    dt = es.dt
    duration = es.duration
    ramp_up_time = es.ramp_up_time
    ramp_down_time = es.ramp_down_time
    efi = es.segment_efield_integrators[3]
    max_potential = 1e3 * (
        efi.displacementX * es.fields[0]["Ex"]
        + efi.displacementY * es.fields[0]["Ey"]
        + efi.displacementZ * es.fields[0]["Ez"]
    )  # from mV to V

    def f_cos(t):
        return max_potential * np.cos(
            2 * np.pi * es.fields[0]["Frequency"] / 1000 * t + es.fields[0]["Phase"]
        )

    t_beforebreakpoint = np.arange(dt / 2, duration + ramp_up_time + ramp_down_time, dt)

    # check the references against the cosine function
    def make_ramp_envelope(t_vec):
        envelope = np.ones(len(t_vec))
        envelope = np.where(t_vec < ramp_up_time, t_vec / ramp_up_time, envelope)
        envelope = np.where(
            t_vec > ramp_up_time + duration,
            1 - (t_vec - (ramp_up_time + duration)) / ramp_down_time,
            envelope,
        )
        return envelope

    ramping_mod = make_ramp_envelope(t_beforebreakpoint)
    npt.assert_allclose(f_cos(t_beforebreakpoint) * ramping_mod, REF_COSINE[1:], atol=1e-6)


REF_CONSTANT = np.array(
    [
        -0.0,
        -0.160722,
        -0.4821677,
        -0.8036128,
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
        -0.843793,
        -0.60271,
        -0.361626,
        -0.120542,
    ]
)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 17},
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
    2. check potential of 1st segment should be 0 (soma),
    and a constant vec for 4th segment including ramp up and down
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    es = stimulus.stimList[0]
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(es.segment_efield_integrators) == total_segments

    soma_seg = cell.soma[0](0.5)
    dend_seg = cell.dend[0](0.25)
    rec_dend = Nd.Vector()
    rec_dend.record(dend_seg.extracellular._ref_e)
    rec_soma = Nd.Vector()
    rec_soma.record(soma_seg.extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()
    npt.assert_allclose(rec_dend, REF_CONSTANT, atol=1e-6)
    npt.assert_allclose(rec_soma, np.zeros(len(rec_dend)))


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 17},
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
    2. check potential of 1st segment should be 0 (soma),
       for 4th segment the sum of the cosine fields and constant fields
    3. check an extracellar mechanism is added to each segment
    4. check the long/unused vectors of ElectrodeSource object are cleaned at the end
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    es = stimulus.stimList[0]
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(es.segment_efield_integrators) == total_segments

    soma_seg = cell.soma[0](0.5)
    dend_seg = cell.dend[0](0.25)
    rec_dend = Nd.Vector()
    rec_dend.record(dend_seg.extracellular._ref_e)
    rec_soma = Nd.Vector()
    rec_soma.record(soma_seg.extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()
    npt.assert_allclose(rec_dend, REF_COSINE + REF_CONSTANT, atol=1e-6)
    npt.assert_allclose(rec_soma, np.zeros(len(rec_dend)))

    assert all(sec.has_membrane("extracellular") for sec in cell.all)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 22},
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
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    dt = stimulus.dt
    delay = stimulus.delay
    npt.assert_approx_equal(delay, 5)

    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    dend_seg = cell.dend[0](0.25)
    rec_dend = Nd.Vector()
    rec_dend.record(dend_seg.extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()
    npt.assert_allclose(
        rec_dend, np.concatenate([np.zeros(int(delay / dt)), REF_COSINE + REF_CONSTANT]), atol=1e-6
    )


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 22},
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
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    dt = stimulus.dt
    delay = stimulus.delay
    npt.assert_approx_equal(delay, 5)

    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    dend_seg = cell.dend[0](0.25)
    rec_dend = Nd.Vector()
    rec_dend.record(dend_seg.extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()
    npt.assert_allclose(
        rec_dend,
        np.concatenate([np.zeros(int(delay / dt)), REF_COSINE + REF_CONSTANT + 2 * REF_CONSTANT]),
        atol=1e-5,
    )


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
