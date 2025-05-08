
import pytest
import numpy as np
import numpy.testing as npt

from tests.conftest import NGV_DIR
from neurodamus import Neurodamus
from neurodamus.ngv import GlioVascularManager, GlutList


def get_Rad(astro_id, manager):
    astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]
    if astrocyte.endfeet is None:
        return []
    return [sec(0.5).vascouplingB.Rad for sec in astrocyte.endfeet]


def get_R0pas(astro_id, manager):
    astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]
    if astrocyte.endfeet is None:
        return []
    return [sec(0.5).vascouplingB.R0pas for sec in astrocyte.endfeet]


def test_glut_list():
    """Test base functionality of glut_list"""

    ll = GlutList(range(5), -1)
    assert ll == [0, 1, 2, 3, 4, -1]
    ll.pop()
    assert ll == [0, 1, 2, 3, -1]
    ll.append(10)
    assert ll == [0, 1, 2, 3, 10, -1]
    ll[2] = 100
    assert ll == [0, 1, 100, 3, 10, -1]
    ll[-1] = -2
    assert ll == [0, 1, 100, 3, 10, -2]
    assert ll.tail == -2
    ll.tail = -3
    assert ll == [0, 1, 100, 3, 10, -3]


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(NGV_DIR),
        "simconfig_file": "simulation_config.json"
    }
], indirect=True)
def test_vasccouplingB_radii(create_tmp_simulation_config_file):
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)

    manager_gliovasc = n.circuits.get_edge_manager("vasculature", "AstrocyteA", GlioVascularManager)
    vasculature_pop = manager_gliovasc._vasculature

    # Compute of vascouplingB.R0Pas ref from vasculature section start and end diameters
    vessel_start_ref = vasculature_pop.get_attribute("start_diameter", 2)
    vessel_end_ref = vasculature_pop.get_attribute("end_diameter", 2)
    R0pas_ref = (vessel_start_ref + vessel_end_ref) / 4
    base_rad_in_vasccouplingBmod = 14.7

    Rad_base = get_Rad(1, manager_gliovasc)[0]
    assert Rad_base == base_rad_in_vasccouplingBmod
    R0pas_old = get_R0pas(1, manager_gliovasc)[0]
    npt.assert_allclose(R0pas_old, R0pas_ref)

    astrocyte = manager_gliovasc._cell_manager.gid2cell[1 + manager_gliovasc._gid_offset]
    Rad_vec = Nd.Vector()
    Rad_vec.record(next(iter(astrocyte.endfeet))(0.5).vascouplingB._ref_Rad)

    Nd.finitialize()
    n.run()

    # Check RingA cells spikes
    spike_gid_ref = np.array([1001, 1002, 1003])
    timestamps_ref = np.array([2.075, 2.075, 2.075])
    ringA_spikes = n._spike_vecs[0]
    timestamps = np.array(ringA_spikes[0])
    spike_gids = np.array(ringA_spikes[1])
    npt.assert_equal(spike_gid_ref, spike_gids)
    npt.assert_allclose(timestamps_ref, timestamps)

    # Check AstrocytesA spikes
    spike_gid_ref = np.array([1, 2])
    timestamps_ref = np.array([5.25, 6.275])
    astrocyteA_spikes = n._spike_vecs[1]
    timestamps = np.array(astrocyteA_spikes[0])
    spike_gids = np.array(astrocyteA_spikes[1])
    npt.assert_equal(spike_gids, spike_gid_ref)
    npt.assert_allclose(timestamps, timestamps_ref)

    # Check Rad variation
    Rad_ref = np.array(
        [14.7, 14.7000011, 14.70000471, 14.70001065, 14.70001896, 14.70002959, 14.70004255,
         14.70005779, 14.7000753,  14.70009505, 14.70011703, 14.70014121, 14.70016757,
         14.70019609, 14.70022675, 14.70025952, 14.70029439, 14.70033134, 14.70037035,
         14.70041139, 14.70045445])
    npt.assert_allclose(Rad_ref, Rad_vec[::20])

    # Check R0pas stability
    R0pas_new = get_R0pas(1, manager_gliovasc)
    npt.assert_allclose(R0pas_new, R0pas_ref)
