
from ..conftest import NGV_DIR
import pytest
import numpy as np
import numpy.testing as npt


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


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(NGV_DIR),
        "simconfig_file": "simulation_config.json"
    }
], indirect=True)
def test_vasccouplingB_radii(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.ngv import GlioVascularManager

    n = Neurodamus(create_tmp_simulation_config_file)

    manager_gliovasc = n.circuits.get_edge_manager("vasculature", "AstrocyteA", GlioVascularManager)
    vasculature_pop = manager_gliovasc._vasculature

    # Compute of vascouplingB.R0Pas ref from vasculature section start and end diameters
    vessel_start_ref = vasculature_pop.get_attribute("start_diameter", 2)
    vessel_end_ref = vasculature_pop.get_attribute("end_diameter", 2)
    R0pas_ref = (vessel_start_ref + vessel_end_ref) / 4
    base_rad_in_vasccouplingBmod = 14.7

    Rad_old = get_Rad(1, manager_gliovasc)[0]
    assert Rad_old == base_rad_in_vasccouplingBmod
    R0pas_old = get_R0pas(1, manager_gliovasc)[0]
    npt.assert_allclose(R0pas_old, R0pas_ref)

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
    timestamps_ref = np.array([5.475, 6.725])
    astrocyteA_spikes = n._spike_vecs[1]
    timestamps = np.array(astrocyteA_spikes[0])
    spike_gids = np.array(astrocyteA_spikes[1])
    npt.assert_equal(spike_gid_ref, spike_gids)
    npt.assert_allclose(timestamps_ref, timestamps)

    # Check Rad variation
    Rad_new = get_Rad(1, manager_gliovasc)[0]
    assert base_rad_in_vasccouplingBmod != Rad_new
    assert 15. > Rad_new > 14.

    # Check R0pas stability
    R0pas_new = get_R0pas(1, manager_gliovasc)
    npt.assert_allclose(R0pas_new, R0pas_ref)
