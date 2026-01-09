import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import RINGTEST_DIR

from neurodamus import Node
from neurodamus.core import MPI
from neurodamus.stimulus_manager import SpatiallyUniformEField

REF_CONSTANT_BIGCELL = np.array(
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

REF_CONSTANT = {0: REF_CONSTANT_BIGCELL, 1: REF_CONSTANT_SMALLCELL}


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
                        "node_set": "RingA",
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
@pytest.mark.mpi(ranks=2)
def test_one_constant_field(create_tmp_simulation_config_file, mpi_ranks):
    """
    A constant field when frequency = 0
    1. check the size of segs_stim_vec, should be applied to all the segments, n_seg
    2. check time_vec of 1st and 4th stimulus, should include ramp_up_time and ramp_down_time
    3. check stim_vec of 1st stimulus should be 0 (soma),
    and a constant vec for 3rd stimlus including ramp up and down
    """
    assert MPI.size == mpi_ranks == 2

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_stimulus()
    local_gids_ref = [[0, 2], [1]]
    local_gids = n.circuits.get_node_manager("RingA").local_nodes.gids(raw_gids=False)
    npt.assert_allclose(local_gids, local_gids_ref[MPI.rank])

    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    gid = local_gids[0]
    cell = n.circuits.get_node_manager("RingA").get_cell(gid)
    cellref = cell.CellRef
    es = stimulus.stimList[gid]
    assert len(es.segs_stim_vec) == sum(sec.nseg for sec in cellref.all)
    duration = stimulus.duration + stimulus.ramp_up_time + stimulus.ramp_down_time
    dt = stimulus.dt
    soma_obj = cellref.soma[0]
    soma_seg_points = cell.segment_global_coords[soma_obj.name()]
    npt.assert_allclose(es.base_position, np.array(soma_seg_points).mean(axis=0))
    seg_stimuli = list(es.segs_stim_vec.values())
    dend_stim_vec = seg_stimuli[3]
    ref_timevec = np.append(np.arange(0, duration + 1, dt), duration)
    npt.assert_allclose(es.time_vec, ref_timevec)
    npt.assert_allclose(dend_stim_vec, REF_CONSTANT[gid], rtol=1e-6)
    n.clear_model()
