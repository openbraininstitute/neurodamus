
import libsonata
import numpy as np
import numpy.testing as npt
import pytest
from mpi4py import MPI

from neurodamus import Neurodamus
from neurodamus.ngv import GlioVascularManager
from tests.conftest import NGV_DIR


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_vascouplingB_attribute(astro_id, manager, attr):
    """
    Retrieve a specified attribute from all endfeet of an astrocyte's vascouplingB mechanism.

    Parameters:
        astro_id (int): The ID of the astrocyte.
        manager: The cell/edge manager containing astrocyte and connectivity data.
        attr (str): The name of the attribute to retrieve (e.g., "Rad", "R0pas").

    Returns:
        list: Attribute values from each endfoot, or an empty list if none exist.
    """
    astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]
    if astrocyte.endfeet is None:
        return []
    return [getattr(sec(0.5).vascouplingB, attr) for sec in astrocyte.endfeet]


def compute_R0pas_from_vasculature_pop(astro_id, manager_gliovasc, vasculature_pop):
    """
    Compute the expected R0pas value for an astrocyte based on afferent vascular diameters.

    Parameters:
        astro_id (int): The ID of the astrocyte.
        manager_gliovasc: The manager handling glio-vascular connections.
        vasculature_pop: The vasculature population object.

    Returns:
        float or ndarray: Average of (start_diameter + end_diameter) / 4 for afferent vessels.
    """
    endfeet = manager_gliovasc._gliovascular.afferent_edges(astro_id-1)
    vasc_node_ids = libsonata.Selection(manager_gliovasc._gliovascular.source_nodes(endfeet))
    d_starts = vasculature_pop.get_attribute("start_diameter", vasc_node_ids)
    d_ends = vasculature_pop.get_attribute("end_diameter", vasc_node_ids)
    return (d_starts+d_ends)/4


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(NGV_DIR),
        "simconfig_file": "simulation_config.json"
    }
], indirect=True)
@pytest.mark.mpi(ranks=1)
def test_vasccouplingB_radii(create_tmp_simulation_config_file, mpi_ranks):
    """
    Test function to validate vascouplingB mechanism attributes and related spike activity.

    This includes:
    - Checking initial Rad values.
    - Computing and verifying R0pas against expected values.
    - Validating spike timings of RingA and AstrocyteA neurons.
    - Ensuring Rad values evolve as expected during simulation.
    - Confirming R0pas remains stable after the simulation.
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)

    astro_ids = list(n.circuits.get_node_manager("AstrocyteA").gid2cell.keys())

    manager_gliovasc = n.circuits.get_edge_manager("vasculature", "AstrocyteA", GlioVascularManager)
    vasculature_pop = manager_gliovasc._vasculature

    R0pas_refs = [compute_R0pas_from_vasculature_pop(astro_id, manager_gliovasc, vasculature_pop)
                  for astro_id in astro_ids]

    for astro_id, R0pas_ref in zip(astro_ids, R0pas_refs):
        # check Rad
        Rads = get_vascouplingB_attribute(astro_id, manager_gliovasc, "Rad")
        npt.assert_allclose(Rads, [14.7]*len(Rads))

        # check R0pas
        R0pas = get_vascouplingB_attribute(astro_id, manager_gliovasc, "R0pas")
        npt.assert_allclose(R0pas, R0pas_ref)

    astrocyte = manager_gliovasc._cell_manager.gid2cell[1 + manager_gliovasc._gid_offset]
    Rad_vec = Nd.Vector()
    Rad_vec.record(next(iter(astrocyte.endfeet))(0.5).vascouplingB._ref_Rad)

    Nd.finitialize()
    n.run()

    # Check RingA cells spikes
    spike_gid_ref = np.array([1001, 1002, 1003, 1004, 1005])
    timestamps_ref = np.array([2.075, 2.075, 2.075, 2.075, 2.075])
    ringA_spikes = n._spike_vecs[0]
    timestamps = np.array(ringA_spikes[0])
    spike_gids = np.array(ringA_spikes[1])
    npt.assert_equal(spike_gid_ref, spike_gids)
    npt.assert_allclose(timestamps_ref, timestamps)

    # Check AstrocytesA spikes
    spike_gid_ref = np.array([1, 2, 3, 4])
    timestamps_ref = np.array([5.475, 6.725, 7.675, 8.775])
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
    R0pas_new = get_vascouplingB_attribute(1, manager_gliovasc, "R0pas")
    npt.assert_allclose(R0pas_new, R0pas_refs[0])


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(NGV_DIR),
        "simconfig_file": "simulation_config.json"
    }
], indirect=True)
@pytest.mark.mpi(ranks=2)
def test_vasccouplingB_radii_mpi(create_tmp_simulation_config_file, mpi_ranks):
    """
    Test function to validate vascouplingB mechanism attributes .
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)

    astro_ids = list(n.circuits.get_node_manager("AstrocyteA").gid2cell.keys())
    manager_gliovasc = n.circuits.get_edge_manager("vasculature", "AstrocyteA", GlioVascularManager)
    vasculature_pop = manager_gliovasc._vasculature

    R0pas_refs = [compute_R0pas_from_vasculature_pop(astro_id, manager_gliovasc, vasculature_pop)
                  for astro_id in astro_ids]

    for astro_id, R0pas_ref in zip(astro_ids, R0pas_refs):
        # check Rad
        Rads = get_vascouplingB_attribute(astro_id, manager_gliovasc, "Rad")
        npt.assert_allclose(Rads, [14.7]*len(Rads))

        # check R0pas
        R0pas = get_vascouplingB_attribute(astro_id, manager_gliovasc, "R0pas")
        npt.assert_allclose(R0pas, R0pas_ref)

    astrocyte = manager_gliovasc._cell_manager.gid2cell[astro_ids[0] + manager_gliovasc._gid_offset]
    Rad_vec = Nd.Vector()
    Rad_vec.record(next(iter(astrocyte.endfeet))(0.5).vascouplingB._ref_Rad)

    Nd.finitialize()
    n.run()

    # Check Rad variation
    if rank == 0:
        Rad_ref = np.array(
            [14.7, 14.7000011, 14.70000471, 14.70001065, 14.70001896, 14.70002959, 14.70004255,
             14.70005779, 14.7000753,  14.70009505, 14.70011703, 14.70014121, 14.70016757,
             14.70019609, 14.70022675, 14.70025952, 14.70029439, 14.70033134, 14.70037035,
             14.70041139, 14.70045445])
    elif rank == 1:
        Rad_ref = np.array(
            [14.7, 14.700001, 14.700005, 14.700011, 14.700019, 14.700029, 14.700042,
             14.700058, 14.700075, 14.700095, 14.700117, 14.700141, 14.700167,
             14.700195, 14.700226, 14.700259, 14.700293, 14.70033, 14.700369,
             14.70041 , 14.700453])
    npt.assert_allclose(Rad_ref, Rad_vec[::20])

    # Check R0pas stability
    R0pas_new = get_vascouplingB_attribute(astro_ids[0], manager_gliovasc, "R0pas")
    npt.assert_allclose(R0pas_new, R0pas_refs[0])
