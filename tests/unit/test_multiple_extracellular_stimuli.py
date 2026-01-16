from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import find_peaks

from tests.conftest import RINGTEST_DIR
from tests.utils import read_ascii_report, record_compartment_reports, write_ascii_reports

from neurodamus import Neurodamus, Node
from neurodamus.core.stimuli import ElectrodeSource
from neurodamus.stimulus_manager import SpatiallyUniformEField

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


def test_combine_time_stim_vectors():  # noqa: PLR0915
    # case 1, overlap, no delay
    t1_vec = np.array([0, 0.5, 1.0, 1.5, 2, 2.5, 2.5])
    stim1_vec = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
    t2_vec = np.array([0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5])
    stim2_vec = np.array([100, 100, 100, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0])
    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=False, is_delay2=False, dt=0.5
    )
    npt.assert_allclose(
        res_time_vec,
        [
            0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            3.5,
        ],
    )
    npt.assert_allclose(res_stim_vec, [110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 100.0, 100.0, 0.0])

    # case 2, overlap, delay
    t1_vec = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
    stim1_vec = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
    t2_vec = np.array([0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0])
    stim2_vec = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0])
    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=True, is_delay2=True, dt=1.0
    )
    npt.assert_allclose(
        res_time_vec,
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            7.0,
        ],
    )
    npt.assert_allclose(res_stim_vec, [0, 10.0, 10.0, 110.0, 110.0, 110.0, 100.0, 100.0, 0.0])

    # case 3, no overlap, t1_vec before t2_vec
    t1_vec = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.2])
    stim1_vec = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0])
    t2_vec = np.array([0.0, 30.0, 30.2, 30.4, 30.6, 30.8, 31.0, 31.0])
    stim2_vec = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0])
    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=False, is_delay2=True, dt=0.2
    )
    npt.assert_allclose(
        res_time_vec, [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.2, 30.0, 30.2, 30.4, 30.6, 30.8, 31, 31]
    )
    npt.assert_allclose(
        res_stim_vec,
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0],
    )

    # case 4, full inclusion, t2 with delay
    t1_vec = np.array([0.0, 1.2, 2.4, 3.6, 4.8, 6, 7.2, 7.2])
    stim1_vec = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
    t2_vec = np.array([0.0, 2.4, 3.6, 4.8, 6, 6])
    stim2_vec = np.array([0.0, 100.0, 100.0, 100.0, 0.0, 0.0])
    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=False, is_delay2=True, dt=1.2
    )
    npt.assert_allclose(
        res_time_vec,
        [
            0,
            1.2,
            2.4,
            3.6,
            4.8,
            6,
            7.2,
            7.2,
        ],
    )
    npt.assert_allclose(res_stim_vec, [0, 10, 110, 110, 110, 10, 10, 0])

    # case 5, no overlap, t2_vec before t1_vec
    t1_vec = np.array([0.0, 30.0, 30.025, 30.05, 30.075, 31.1, 31.1])
    stim1_vec = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0])

    t2_vec = np.array([0.0, 11.0, 11.025, 11.05, 11.075, 11.1, 11.1])
    stim2_vec = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0])

    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=True, is_delay2=True, dt=0.025
    )
    npt.assert_allclose(
        res_time_vec,
        [0.0, 11, 11.025, 11.05, 11.075, 11.1, 11.1, 30.0, 30.025, 30.05, 30.075, 31.1, 31.1],
    )
    npt.assert_allclose(res_stim_vec, [0.0, 10, 10, 10, 10, 10, 10, 0, 100, 100, 100, 100, 100, 0])

    # case 6, no overlap, t2_vec before t1_vec, with delay
    t1_vec = np.array([0, 30.0, 30.2, 30.4, 30.6, 30.8, 30.8])
    stim1_vec = np.array([0, 100, 100, 100, 100, 100, 0])

    t2_vec = np.array([0.0, 10.0, 10.2, 10.4, 10.4])
    stim2_vec = np.array([0.0, 10, 10, 10, 0])

    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=True, is_delay2=True, dt=0.2
    )
    npt.assert_allclose(
        res_time_vec, [0.0, 10.0, 10.2, 10.4, 10.4, 30.0, 30.2, 30.4, 30.6, 30.8, 30.8]
    )
    npt.assert_allclose(res_stim_vec, [0, 10, 10, 10, 0, 100, 100, 100, 100, 100, 0])

    # case 7 t1 and t2 is continuous
    # t1 before t2
    t1_vec = np.array([0, 9.0, 9.2, 9.4, 9.6, 9.8, 9.8])
    stim1_vec = np.array([0, 100, 100, 100, 100, 100, 0])
    t2_vec = np.array([0.0, 10.0, 10.2, 10.4, 10.4])
    stim2_vec = np.array([0.0, 10, 10, 10, 0])
    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=True, is_delay2=True, dt=0.2
    )
    npt.assert_allclose(res_time_vec, [0.0, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.4])
    npt.assert_allclose(res_stim_vec, [0, 100, 100, 100, 100, 100, 10, 10, 10, 0])

    # t2 before t1
    t1_vec = np.array([0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.5])
    stim1_vec = np.array([0, 100, 100, 100, 100, 100, 0])
    t2_vec = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.25])
    stim2_vec = np.array([10, 10, 10, 10, 10, 10, 0])
    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=True, is_delay2=False, dt=0.25
    )
    npt.assert_allclose(
        res_time_vec, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.5]
    )
    npt.assert_allclose(
        res_stim_vec, [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0]
    )

    # case 8, t1 = t2
    # with delay
    t1_vec = t2_vec = np.array([0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 70.0])
    stim1_vec = stim2_vec = np.array([0, 100.0, 100.0, 100, 100, 100, 100, 100, 0])

    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=True, is_delay2=True, dt=10.0
    )
    npt.assert_allclose(res_time_vec, t1_vec)
    npt.assert_allclose(res_stim_vec, np.add(stim1_vec, stim2_vec))

    # without delay
    t1_vec = t2_vec = np.array([0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 70.0])
    stim1_vec = stim2_vec = np.array([0, 100.0, 100.0, 100, 100, 100, 100, 100, 0])
    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=False, is_delay2=False, dt=10.0
    )
    npt.assert_allclose(res_time_vec, t1_vec)
    npt.assert_allclose(res_stim_vec, np.add(stim1_vec, stim2_vec))

    # case 9, one time point in t1 and t2, with delay
    t1_vec = np.array([0.0, 10.025, 10.025])
    stim1_vec = np.array([0.0, 10.0, 0.0])
    t2_vec = np.array([0.0, 10.075, 10.075])
    stim2_vec = np.array([0.0, 100.0, 0.0])
    res_time_vec, res_stim_vec = ElectrodeSource.combine_time_stim_vectors(
        t1_vec, stim1_vec, t2_vec, stim2_vec, is_delay1=True, is_delay2=True, dt=0.025
    )
    npt.assert_allclose(res_time_vec, [0, 10.025, 10.025, 10.075, 10.075])
    npt.assert_allclose(res_stim_vec, [0, 10, 0, 100, 0])


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
def test_two_stimulus_blocks(create_tmp_simulation_config_file):
    """
    Two stimulus blocks, one contains a cosine field and the other contains a constant field,
    they should be summed before applying
    1. check their stimulus managers share the same SpatiallyUniformEField instance (singleton)
    2. check the size of segs_stim_vec, should be applied to all the segments, n_seg
    3. check time_vec of 1st and 4th stimulus, should include ramp_up_time and ramp_down_time
    4. check stim_vec of 1st stimulus should be 0 (soma),
       for 3rd stimlus the sum of the cosine fields and constant fields
    5. check an extracellar mechanism is added to each segment
    """

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    assert len(n._stim_manager._stimulus) == 1
    stimulus = n._stim_manager._stimulus[0]
    assert stimulus == SpatiallyUniformEField._instance
    cell_manager = n.circuits.get_node_manager("RingA")
    cell = cell_manager.get_cellref(0)
    assert list(stimulus.stimList.keys()) == [0]
    es = stimulus.stimList[0]
    assert isinstance(es, ElectrodeSource)
    total_segments = sum(sec.nseg for sec in cell.all)
    assert len(es.segs_stim_vec) == total_segments
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    ref_stimvec = np.zeros(len(ref_timevec))
    seg_stimuli = list(es.segs_stim_vec.values())
    soma_stim_vec = seg_stimuli[0]
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(soma_stim_vec, ref_stimvec)
    seg_stim_vec = seg_stimuli[3]
    npt.assert_allclose(seg_stim_vec, REF_COSINE + REF_CONSTANT, rtol=1e-5)

    assert all(sec.has_membrane("extracellular") for sec in cell.all)

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
    Check the combination of two stimulus blocks, one with delay.
    The original delay 4.5 is not divisible by dt 1.0, rounded up to the next time point.
    """

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    stimulus = n._stim_manager._stimulus[0]
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    captured = capsys.readouterr()
    warning = "[WARNING] SpatiallyUniformEField delay 4.5 is not divisible by dt 1.0, rounded up to the next time point 5.0"
    assert warning in captured.out
    delay = stimulus.delay
    npt.assert_approx_equal(delay, 5)
    ref_timevec = [0, *np.arange(delay, delay + duration + 1, dt), delay + duration]
    ref_stimvec = np.zeros(len(ref_timevec))
    es = stimulus.stimList[0]
    seg_stimuli = list(es.segs_stim_vec.values())
    soma_stim_vec = seg_stimuli[0]
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(soma_stim_vec, ref_stimvec)
    dend_stim_vec = seg_stimuli[3]
    npt.assert_allclose(dend_stim_vec, np.append(0, REF_COSINE + REF_CONSTANT), rtol=1e-5)
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
    ref_timevec = [0, *np.arange(delay, delay + duration + 1, dt), delay + duration]
    ref_stimvec = np.zeros(len(ref_timevec))
    es = stimulus.stimList[0]
    seg_stimuli = list(es.segs_stim_vec.values())
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
    ref_timevec = [0, *np.arange(delay, delay + duration + 1, dt), delay + duration]
    # cell 0
    es0 = stimulus.stimList[0]
    seg_stimuli = list(es0.segs_stim_vec.values())
    dend_stim_vec = seg_stimuli[3]
    npt.assert_allclose(es0.time_vec, ref_timevec)
    npt.assert_allclose(dend_stim_vec, np.append(0, REF_COSINE + REF_CONSTANT), rtol=1e-5)

    # cell 1
    es1 = stimulus.stimList[1]
    seg_stimuli = list(es1.segs_stim_vec.values())
    dend_stim_vec = seg_stimuli[3]
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
        17.0,
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
    seg_stimuli = list(es.segs_stim_vec.values())
    dend_stim_vec = seg_stimuli[3]
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
    write_ascii_reports(ascii_recorders, n._run_conf["OutputRoot"])

    # Read ASCII reports
    report = Path(n._run_conf["OutputRoot"]) / ("compartment_v.txt")
    assert report.exists()
    data = read_ascii_report(report)
    cell0_voltage_vec = [vec[3] for vec in data if vec[0] == 0]
    assert len(cell0_voltage_vec) == 21 * 23  # 21 time steps * 23 compartments
    peaks_pos = find_peaks(cell0_voltage_vec, prominence=1)[0]
    np.testing.assert_allclose(peaks_pos, ref_peak)
    n.clear_model()  # clear up the reporting vector, required for the next run.
