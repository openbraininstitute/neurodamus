import numpy as np
import numpy.testing as npt
import pytest
from neuron import h

from tests.conftest import RINGTEST_DIR

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
    1. check the size of stimList, should be applied to all the segments, n_seg
    2. check time_vec of 1st and 3rd stimulus, no ramp_up_time and ramp_down_time
    3. check the base point of the stimli = the mean of the soma segment points
    4. check stim_vec of 1st stimulus should be 0 (soma), and a sin wave for 4th stimlus (dendrite)
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cellref = cell_manager.get_cell(0)
    cellref = cell.CellRef
    total_segments = sum(sec.nseg for sec in cellref.all)
    assert len(stimulus.stimList) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    soma_signal_source = stimulus.stimList[0]
    soma_obj = cellref.soma[0]
    soma_seg_points = cell.all_segment_points[soma_obj.name()]
    assert isinstance(soma_signal_source, ElectrodeSource)
    npt.assert_allclose(soma_signal_source.base_position, np.array(soma_seg_points).mean(axis=0))
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    npt.assert_allclose(soma_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(soma_signal_source.stim_vec, ref_stimvec)
    seg_signal_source = stimulus.stimList[3]
    ref_stimvec = [
        0.860194,
        0.695911,
        0.265814,
        -0.265814,
        -0.695911,
        -0.860194,
        -0.695911,
        -0.265814,
        0.265814,
        0.695911,
        0.860194,
        0.0,
    ]
    npt.assert_allclose(seg_signal_source.time_vec, ref_timevec)
    npt.assert_allclose(seg_signal_source.stim_vec, ref_stimvec, rtol=1e-5)
    n.clear_model()


REF_COSINE = np.array(
    [
        0.0,
        0.347956,
        0.265814,
        -0.265814,
        -0.695911,
        -0.860194,
        -0.695911,
        -0.265814,
        0.265814,
        0.695911,
        0.860194,
        0.695911,
        0.265814,
        -0.265814,
        -0.695911,
        -0.573462,
        -0.23197,
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
    1. check the size of stimList, should be applied to all the segments, n_seg
    2. check time_vec of 1st and 4th stimulus, should include ramp_up_time and ramp_down_time
    3. check stim_vec of 1st stimulus should be 0 (soma),
    and a cosine wave with 3 ramp up steps and 4 ramp down steps
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
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
    npt.assert_allclose(seg_signal_source.stim_vec, REF_COSINE, rtol=1e-5)
    n.clear_model()


REF_CONSTANT = np.array(
    [
        0.0,
        0.860194,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.720387,
        1.146925,
        0.573462,
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
    2. check time_vec of 1st and 4th stimulus, should include ramp_up_time and ramp_down_time
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
    cell = cell_manager.get_cellref(0)
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
    npt.assert_allclose(seg_signal_source.stim_vec, REF_CONSTANT, rtol=1e-6)
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
    1. check the size of stimList, should be applied to all the segments, n_seg
    2. check time_vec of 1st and 4th stimulus, should include ramp_up_time and ramp_down_time
    3. check stim_vec of 1st stimulus should be 0 (soma),
    and a constant vec for 3rd stimlus the sum of the cosine fields and constant fields
    4. check an extracellar mechanism is added to each segment
    """

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
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
    npt.assert_allclose(seg_signal_source.stim_vec, REF_COSINE + REF_CONSTANT, rtol=1e-5)
    for sec in cell.all:
        for seg in sec:
            assert hasattr(seg, "extracellular")
    n.clear_model()
