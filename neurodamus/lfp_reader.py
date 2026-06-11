"""LFP electrodes file reader for neurodamus."""

import logging

import h5py
import numpy as np

from .core import NeuronWrapper as Nd
from .core.configuration import ConfigurationError


class LFPFileReader:
    """Driver for an LFP electrodes HDF5 file.

    Opens the file and validates basic structure on construction.
    Provides lazy per-gid access to electrode scaling factors via h5py
    (no bulk preloading).
    """

    def __init__(self, filepath: str) -> None:
        """Open the electrodes file and validate its structure."""
        try:
            self._file = h5py.File(filepath, "r")
        except OSError as e:
            raise ConfigurationError(f"Error opening LFP electrodes file: {filepath}") from e
        self._filepath = filepath
        self._validate()

    def _validate(self) -> None:
        """Check that the file has the required top-level 'electrodes' group."""
        if "electrodes" not in self._file:
            raise ConfigurationError(
                f"LFP electrodes file '{self._filepath}' is missing the 'electrodes' group."
            )

    def get_number_electrodes(self, gid: int, population_info: tuple[str, int]) -> int:
        """Return the number of electrodes for a given gid.

        Args:
            gid: The global cell identifier.
            population_info: (population_name, population_offset) from
                GlobalCellManager.getPopulationInfo().
        """
        population_name, offset = population_info
        node_id = gid - offset
        try:
            subset = self._get_node_subsets(node_id, population_name)
            return subset.shape[1]
        except (KeyError, IndexError) as e:
            logging.warning(
                "Node id %d missing in '%s' for population %s: %s",
                node_id,
                self._filepath,
                population_name,
                e,
            )
            return 0

    def get_factors(self, gid: int, population_info: tuple[str, int]):
        """Read LFP scaling factors for a given gid as a flat Nd.Vector.

        Args:
            gid: The global cell identifier.
            population_info: (population_name, population_offset) from
                GlobalCellManager.getPopulationInfo().

        Returns:
            Nd.Vector with concatenated per-compartment electrode factors,
            or an empty vector if the gid is not found.
        """
        population_name, offset = population_info
        node_id = gid - offset
        try:
            subset = self._get_node_subsets(node_id, population_name)
            factors = Nd.Vector()
            for electrode_factors in subset:
                factors.append(Nd.Vector(electrode_factors))
        except (KeyError, IndexError) as e:
            logging.warning(
                "Node id %d missing in '%s' for population %s: %s",
                node_id,
                self._filepath,
                population_name,
                e,
            )
            return Nd.Vector()
        else:
            return factors

    def get_scaling_matrix(self, gid: int, population_info: tuple[str, int]) -> np.ndarray | None:
        """Read LFP scaling factors for a given gid as a numpy array.

        Args:
            gid: The global cell identifier.
            population_info: (population_name, population_offset) from
                GlobalCellManager.getPopulationInfo().

        Returns:
            2D array of shape (n_compartments, n_electrodes), or None if not found.
        """
        population_name, offset = population_info
        node_id = gid - offset
        try:
            return self._get_node_subsets(node_id, population_name)
        except (KeyError, IndexError) as e:
            logging.warning(
                "Node id %d missing in '%s' for population %s: %s",
                node_id,
                self._filepath,
                population_name,
                e,
            )
            return None

    def _get_node_subsets(self, node_id: int, population_name: str) -> np.ndarray:
        """Retrieve the scaling factor matrix for a single node.

        Args:
            node_id: The 0-based SONATA node identifier.
            population_name: The population group name in the HDF5 file.

        Returns:
            2D array of shape (n_compartments, n_electrodes).
        """
        node_ids = self._file[population_name]["node_ids"]
        index = np.where(np.array(node_ids) == node_id)[0][0]
        offsets = self._file[population_name]["offsets"]
        scaling = self._file["electrodes"][population_name]["scaling_factors"]
        return scaling[offsets[index] : offsets[index + 1], :]

    def close(self) -> None:
        self._file.close()

    def __del__(self) -> None:
        if hasattr(self, "_file"):
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_) -> None:
        self.close()
