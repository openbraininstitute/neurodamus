import logging

import numpy as np

from .core import NeuronWrapper as Nd


class LFPManager:
    """Class handling the Online Local Field Potential (LFP) functionality.

    This class is designed to manage the configuration and retrieval of Online
    Local Field Potential (LFP) data used in large-scale neural simulations.
    LFPs represent the aggregate extracellular electrical activity recorded by
    electrodes, reflecting the synchronized activity of nearby neuronal
    populations.

    LFP data in this context is stored in HDF5 files, which must include
    information on:
        - Electrode scaling factors (per compartment contribution to each electrode)
        - Node IDs (cell identifiers contributing to the signal)
        - Offsets (to index subsets of the data per neuron)
    """

    @staticmethod
    def get_sonata_node_id(gid, population_info):
        return population_info[0], gid - population_info[1]

    @staticmethod
    def _get_node_id_subsets(lfp_file, node_id, population_name):
        node_ids = lfp_file[population_name]["node_ids"]
        index = np.where(np.array(node_ids) == node_id)[0][0]
        offsets_dataset = lfp_file[population_name]["offsets"]
        electrodes_dataset = lfp_file["electrodes"][population_name]["scaling_factors"]
        index_low = offsets_dataset[index]
        index_high = offsets_dataset[index + 1]
        return electrodes_dataset[index_low:index_high, :]

    @staticmethod
    def read_lfp_factors(lfp_file, gid, population_info):
        """Reads the LFP factors for a specific gid from an open HDF5 file.

        Args:
            lfp_file: An open h5py.File with electrode scaling factors.
            gid (int): The unique cell identifier.
            population_info (Pair(str, int)): ("population_name", population_offset)

        Returns:
            Nd.Vector: A vector containing the LFP factors for the specified gid.
        """
        scalar_factors = Nd.Vector()
        try:
            population_name, node_id = LFPManager.get_sonata_node_id(gid, population_info)
            subset_data = LFPManager._get_node_id_subsets(lfp_file, node_id, population_name)
            for electrode_factors in subset_data:
                scalar_factors.append(Nd.Vector(electrode_factors))
        except (KeyError, IndexError) as e:
            msg = (
                f"Node id {node_id} missing in the electrodes file "
                f"for population {population_name}: {e!s}"
            )
            logging.warning(msg)
        return scalar_factors

    @staticmethod
    def get_number_electrodes(lfp_file, gid, population_info):
        """Get number of electrodes for a given gid."""
        num_electrodes = 0
        try:
            population_name, node_id = LFPManager.get_sonata_node_id(gid, population_info)
            subset_data = LFPManager._get_node_id_subsets(lfp_file, node_id, population_name)
            num_electrodes = subset_data.shape[1]
        except (KeyError, IndexError) as e:
            msg = (
                f"Node id {node_id} missing in the electrodes file "
                f"for population {population_name}: {e!s}"
            )
            logging.warning(msg)
        return num_electrodes
