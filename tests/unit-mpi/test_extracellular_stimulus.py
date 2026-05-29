import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import RINGTEST_DIR
from tests.utils import get_expected_extracellular_potentials

from neurodamus import Neurodamus
from neurodamus.core import MPI
from neurodamus.stimulus_manager import SpatiallyUniformEField


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
    check the constant field in each mpi processes, cell 0 in rank 0 and cell 1 in rank1
    1. check the size of segment_potentials, should be applied to all the segments, n_seg
    2. check potential of the 4th segment, should be a constant vec including ramp up and down
    3. check all segment has extracellular mechanism inserted
    4. check the long/unused vectors of ElectrodeSource object are cleaned at the end
    """
    from neurodamus.core import NeuronWrapper as Nd

    assert MPI.size == mpi_ranks == 2

    n = Neurodamus(create_tmp_simulation_config_file)
    local_gids_ref = [[0, 2], [1]]
    local_gids = n.circuits.get_node_manager("RingA").local_nodes.gids(raw_gids=False)
    npt.assert_allclose(local_gids, local_gids_ref[MPI.rank])

    stimulus = n._stim_manager._stimulus[0]
    assert isinstance(stimulus, SpatiallyUniformEField)
    gid = local_gids[0]
    cellref = n.circuits.get_node_manager("RingA").get_cellref(gid)
    es = stimulus.stimList[gid]
    assert len(es.segment_efield_integrators) == sum(sec.nseg for sec in cellref.all)

    dend_seg = cellref.dend[0](0.25)
    rec_dend = Nd.Vector()
    rec_dend.record(dend_seg.extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    assert len(es.efields) == 1
    tot_tvec = np.concatenate([[0], np.arange(Nd.dt / 2, Nd.tstop, Nd.dt)])
    es = stimulus.stimList[gid]
    efi = es.segment_efield_integrators[3] if gid == 0 else es.segment_efield_integrators[1]
    ref = get_expected_extracellular_potentials(tot_tvec, efi, es.efields)
    npt.assert_allclose(rec_dend, ref)

    assert all(sec.has_membrane("extracellular") for sec in cellref.all)
