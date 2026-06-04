import logging

import h5py
import numpy as np

from .core import NeuronWrapper as Nd
from .core.configuration import ConfigurationError


class LFPFileReader:
    """Driver for an LFP electrodes HDF5 file.

    Opens the file and validates basic structure on construction.
    Provides lazy per-gid access to electrode scaling factors via h5py
    (no bulk preloading). Use as a context manager or call close() when done.
    """

    def __init__(self, filepath):
        try:
            self._file = h5py.File(filepath, "r")
        except OSError as e:
            raise ConfigurationError(
                f"Error opening LFP electrodes file: {filepath}"
            ) from e
        self._filepath = filepath
        self._validate()

    def _validate(self):
        if "electrodes" not in self._file:
            raise ConfigurationError(
                f"LFP electrodes file '{self._filepath}' is missing the "
                "'electrodes' group."
            )

    def get_number_electrodes(self, gid, population_info):
        """Get number of electrodes for a given gid in a population.

        Args:
            gid: The global cell identifier.
            population_info: Tuple of (population_name, population_offset)
                as returned by GlobalCellManager.getPopulationInfo().
        """
        population_name, offset = population_info
        node_id = gid - offset
        try:
            subset = self._get_node_subsets(node_id, population_name)
            return subset.shape[1]
        except (KeyError, IndexError) as e:
            logging.warning(
                "Node id %d missing in '%s' for population %s: %s",
                node_id, self._filepath, population_name, e,
            )
            return 0

    def get_factors(self, gid, population_info):
        """Read LFP scaling factors for a given gid as a flat Nd.Vector.

        Args:
            gid: The global cell identifier.
            population_info: Tuple of (population_name, population_offset)
                as returned by GlobalCellManager.getPopulationInfo().

        Returns an empty vector if the node_id is not found.
        """
        population_name, offset = population_info
        node_id = gid - offset
        try:
            subset = self._get_node_subsets(node_id, population_name)
            factors = Nd.Vector()
            for electrode_factors in subset:
                factors.append(Nd.Vector(electrode_factors))
            return factors
        except (KeyError, IndexError) as e:
            logging.warning(
                "Node id %d missing in '%s' for population %s: %s",
                node_id, self._filepath, population_name, e,
            )
            return Nd.Vector()

    def _get_node_subsets(self, node_id, population_name):
        node_ids = self._file[population_name]["node_ids"]
        index = np.where(np.array(node_ids) == node_id)[0][0]
        offsets = self._file[population_name]["offsets"]
        scaling = self._file["electrodes"][population_name]["scaling_factors"]
        return scaling[offsets[index]:offsets[index + 1], :]

    def close(self):
        self._file.close()

    def __del__(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
