"""Stimulus implementation where incoming synaptic events are replayed for a single gid"""

from pathlib import Path

import libsonata

from .core.configuration import ConfigurationError
from .utils.multimap import GroupedMultiMap
from .utils.timeit import timeit


class SpikeManager:
    """Holds and manages gid spike time information, specially for Replay.

    A SynapseReplay stim can be used for a single gid that has all the synapses instantiated.
    Given a spikes file from a previous run, this object uses a NetStim object to retrigger
    the synapses at the appropriate time as though the presynaptic cells were present and active.

    Internally the spikes are stored in a :py:class:`neurodamus.utils.multimap.GroupedMultiMap`
    """

    _ascii_spike_dtype = [("time", "double"), ("gid", "uint32")]

    @timeit(name="Replay init")
    def __init__(self, spike_filename, delay=0, population=None):
        """Constructor for SynapseReplay.

        Args:
            spike_filename: path to spike out file.
                if ext is .bin, interpret as binary file; otherwise, interpret as ascii
            delay: delay to apply to spike times
        """
        self._gid_fire_events = None
        # Nd.distributedSpikes = 0  # Wonder the effects of this
        self.open_spike_file(spike_filename, delay, population)

    def open_spike_file(self, filename, delay, population=None):
        """Opens a given spike file.

        Args:
            filename: path to spike out file. Interpret as binary or ascii according to extension
            delay: delay to apply to spike times
        """
        if Path(filename).suffix == (".h5"):
            tvec, gidvec = self._read_spikes_sonata(filename, population)
        else:
            raise ConfigurationError("Spikes input should be a SONATA h5 file")
        if delay:
            tvec += delay

        self._store_events(tvec, gidvec)

    @classmethod
    def _read_spikes_sonata(cls, filename, population):
        spikes_file = libsonata.SpikeReader(filename)
        if population not in spikes_file.get_population_names():
            raise MissingSpikesPopulationError("Spikes population not found: " + population)
        spikes = spikes_file[population]
        spike_dict = spikes.get_dict()
        return spike_dict["timestamps"], spike_dict["node_ids"] + 1

    def _store_events(self, tvec, gidvec):
        """Stores the events in the _gid_fire_events GroupedMultiMap.

        tvec and gidvec arguments should be numpy arrays
        """
        spike_map = GroupedMultiMap(gidvec, tvec)
        if self._gid_fire_events is None:
            self._gid_fire_events = spike_map
        else:
            self._gid_fire_events += spike_map

    def __len__(self):
        return len(self._gid_fire_events)

    def __getitem__(self, gid):
        return self._gid_fire_events.get(gid)

    def __contains__(self, gid):
        return gid in self._gid_fire_events

    def get_map(self):
        """Returns the :py:class:`GroupedMultiMap` with all the spikes."""
        return self._gid_fire_events

    def filter_map(self, pre_gids):
        """Returns a raw dict of pre_gid->spikes for the given pre gids."""
        return {key: self._gid_fire_events[key] for key in pre_gids}


class MissingSpikesPopulationError(Exception):
    """An exception triggered when a given node population is not found, we may want to handle"""
