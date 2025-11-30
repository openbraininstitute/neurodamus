import numpy as np
import numpy.testing as npt
import pytest
from neuron import h

from neurodamus import Node
from neurodamus.core.stimuli import ElectrodeSource
from neurodamus.stimulus_manager import SpatiallyUniformEField


def test_apply_ramp():
    """Test the function apply_ramp"""
    dt = 1
    ref_up_time = 5  # 5 time steps from 0
    ref_down_time = 3  # 3 time steps to 0

    stimulus = ElectrodeSource(0, 100, [], ref_up_time, ref_down_time, dt, base_position=[0, 0, 0])
    stim_vec = h.Vector(range(1, 11))
    assert np.isclose(stimulus.ramp_up_time, ref_up_time)
    assert np.isclose(stimulus.ramp_down_time, ref_down_time)
    assert np.isclose(stimulus.dt, dt)
    stimulus.apply_ramp(stim_vec, stimulus.dt)
    assert np.allclose(stim_vec.as_numpy(), [0, 0.5, 1.5, 3, 5, 6, 7, 8, 4.5, 0])


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "run": {"dt": 1},
                "inputs": {
                    "one_sin_efield": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_oneCell",
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
def test_one_sin_field_noramp(create_tmp_simulation_config_file):
    """
    One sinusoid field without ramp
    1. check the size of stimList, should be applied to all the segments, n_seg
    2. check time_vec of 1st and 3rd stimulus, no ramp_up_time and ramp_down_time
    3. check stim_vec of 1st stimulus should be 0 (soma), and a sin wave for 3rd stimlus
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(1)
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(stimulus.stimList) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    soma_signal_source = stimulus.stimList[0]
    assert isinstance(soma_signal_source, ElectrodeSource)
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    npt.assert_allclose(soma_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(soma_signal_source.stim_vec, ref_stimvec)
    seg_signal_source = stimulus.stimList[3]
    ref_stimvec = [
        0,
        6.424959,
        10.39580,
        10.39580,
        6.424959,
        0,
        -6.424959,
        -10.39580,
        -10.39580,
        -6.424959,
        0,
        0,
    ]
    npt.assert_allclose(seg_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(seg_signal_source.stim_vec, ref_stimvec, atol=1e-14, rtol=1e-6)
    n.clear_model()


REF_SIN = np.array(
    [
        0,
        6.424959 / 2,
        10.39580,
        10.39580,
        6.424959,
        0.0,
        -6.424959,
        -10.39580,
        -10.39580,
        -6.424959,
        0.0,
        6.424959,
        10.39580,
        10.39580,
        6.424959,
        0,
        -6.424959 / 3,
        0,
        0,
    ]
)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "run": {"dt": 1},
                "inputs": {
                    "one_sin_efield": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_oneCell",
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
def test_one_sin_field_withramp(create_tmp_simulation_config_file):
    """
    A sinusoid field with ramp up and down
    1. check the size of stimList, should be applied to all the segments, n_seg
    2. check time_vec of 1st and 3rd stimulus, should include ramp_up_time and ramp_down_time
    3. check stim_vec of 1st stimulus should be 0 (soma),
    and a sin wave with 3 ramp up steps and 4 ramp down steps
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(1)
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(stimulus.stimList) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    soma_signal_source = stimulus.stimList[0]
    assert isinstance(soma_signal_source, ElectrodeSource)
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    npt.assert_allclose(soma_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(soma_signal_source.stim_vec, ref_stimvec)
    seg_signal_source = stimulus.stimList[3]
    npt.assert_allclose(seg_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(seg_signal_source.stim_vec, REF_SIN, atol=1e-14, rtol=1e-6)
    n.clear_model()


REF_CONSTANT = np.array(
    [
        0.0,
        10.930793,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        21.861587,
        14.574391,
        7.287196,
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
                "run": {"dt": 1},
                "inputs": {
                    "one_sin_efield": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_oneCell",
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
    1. check the size of stimList, should be applied to all the segments, n_seg
    2. check time_vec of 1st and 3rd stimulus, should include ramp_up_time and ramp_down_time
    3. check stim_vec of 1st stimulus should be 0 (soma),
    and a constant vec for 3rd stimlus including ramp up and down
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(1)
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(stimulus.stimList) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    soma_signal_source = stimulus.stimList[0]
    assert isinstance(soma_signal_source, ElectrodeSource)
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    npt.assert_allclose(soma_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(soma_signal_source.stim_vec, ref_stimvec)
    seg_signal_source = stimulus.stimList[3]
    npt.assert_allclose(seg_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(seg_signal_source.stim_vec, REF_CONSTANT)
    n.clear_model()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "run": {"dt": 1},
                "inputs": {
                    "ex_efields": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_oneCell",
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
    Two fields that should be summed together sin + constant fields
    1. check the size of stimList, should be applied to all the segments, n_seg
    2. check time_vec of 1st and 3rd stimulus, should include ramp_up_time and ramp_down_time
    3. check stim_vec of 1st stimulus should be 0 (soma),
    and a constant vec for 3rd stimlus the sum of the sin fields and constant fields
    4. check an extracellar mechanism is added to each segment
    """

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(1)
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(stimulus.stimList) == total_segments
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, ElectrodeSource)
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    soma_signal_source = stimulus.stimList[0]
    npt.assert_allclose(soma_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(soma_signal_source.stim_vec, ref_stimvec)
    seg_signal_source = stimulus.stimList[3]
    npt.assert_allclose(seg_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(seg_signal_source.stim_vec, REF_SIN + REF_CONSTANT, atol=3e-6)
    for sec in cell.all:
        for seg in sec:
            assert hasattr(seg, "extracellular")
    n.clear_model()
