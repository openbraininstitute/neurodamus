"""Module implementing interfaces to the several synapse readers (eg.: synapsetool, Hdf5Reader)"""

from __future__ import annotations

import logging

import libsonata
import numpy as np

from neurodamus.core import NeuronWrapper as Nd, ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_verbose
from neurodamus.utils.pyutils import gen_ranges


class SynapseParameters:
    """Synapse parameter names and dtypes for numpy recarrays
    following the SONATA specification.

    For detailed info on the parameters, see:
    https://sonata-extension.readthedocs.io/en/latest/sonata_tech.html#edge-file
    """

    _fields = {
        "sgid": np.int64,
        "delay": np.float64,
        "isec": np.int32,
        "ipt": np.int32,
        "offset": np.float64,
        "weight": np.float64,
        "U": np.float64,
        "D": np.float64,
        "F": np.float64,
        "DTC": np.float64,
        "synType": np.int32,
        "nrrp": np.int32,
        "u_hill_coefficient": np.float64,
        "conductance_ratio": np.float64,
        "maskValue": np.float64,
        "location": np.float64,
    }

    _optional = {"u_hill_coefficient": 0.0, "conductance_ratio": -1.0}
    _reserved = {"maskValue": -1.0, "location": 0.5}

    @classmethod
    def all_fields(cls):
        """Return all defined field names."""
        return set(cls._fields.keys())

    @classmethod
    def load_fields(cls):
        """Return all fields except reserved ones."""
        return cls.all_fields() - set(cls._reserved.keys())

    @classmethod
    def dtype(cls, extra_fields=None):
        """Return dtype including optional extra fields (all float64)."""
        fields = list(cls._fields.items())

        if extra_fields is not None:
            fields.extend((name, np.float64) for name in extra_fields)

        return np.dtype(fields)

    @classmethod
    def fields(cls, exclude: set = (), with_translation: dict | None = None):
        """Return list of fields with optional exclusion and translation.

        Returns list of tuples (field_name, translated_name or is_optional).
        """
        optional_keys = cls._optional.keys()
        fields = cls.load_fields() - exclude if exclude else cls.load_fields()

        if with_translation:
            return [(f, with_translation.get(f, f), f in optional_keys) for f in fields]
        return [(f, f in optional_keys) for f in fields]

    @staticmethod
    def _patch_delay_fp_inaccuracies(records):
        """Round 'delay' to nearest multiple of Nd.dt to fix fp inaccuracies."""
        if len(records) == 0 or "delay" not in records.dtype.names:
            return
        dt = Nd.dt
        records.delay = (records.delay / dt + 1e-5).astype("i4") * dt

    @staticmethod
    def _constrained_hill(K_half, y):  # noqa: N803
        """Constrained Hill function for scaling synaptic parameters.

        Note: it is iused only in scale_U_param. It is its own function
        because it deserves to be tested separately.
        """
        K4 = K_half**4
        y4 = y**4
        return (K4 + 16) / 16 * y4 / (K4 + y4)

    @staticmethod
    def _patch_scale_U_param(syn_params, extra_cellular_calcium, extra_scale_vars):
        """Scale 'U' and other vars using constrained Hill function based on
        extracellular calcium.
        """
        if len(syn_params) == 0 or extra_cellular_calcium is None:
            return

        scale_factors = SynapseParameters._constrained_hill(
            syn_params.u_hill_coefficient, extra_cellular_calcium
        )
        syn_params.U *= scale_factors
        for var in extra_scale_vars:
            syn_params[var] *= scale_factors

    @classmethod
    def make_synapse_parameters_array(
        cls,
        data: dict,
        extra_fields: list[str],
        extra_cellular_calcium: float | None,
        extra_scale_vars: list[str],
    ):
        """Create a recarray from data with optional extra fields and apply patches."""
        if not data:
            return np.recarray(0, dtype=cls.dtype(extra_fields=None))
        edge_count = len(next(iter(data.values())))
        arr = np.recarray(edge_count, dtype=cls.dtype(extra_fields=extra_fields))

        for name in arr.dtype.names:
            if name in cls._reserved:
                arr[name] = cls._reserved[name]
            elif name in cls._optional:
                arr[name] = data.get(name, cls._optional[name])
            elif name in data:
                arr[name] = data[name]
            else:
                raise AttributeError(f"Missing mandatory attribute {name} in the SONATA edge file")

        cls._patch_delay_fp_inaccuracies(arr)
        cls._patch_scale_U_param(arr, extra_cellular_calcium, extra_scale_vars)

        return arr


class SonataReader:
    """Reader for SONATA edge files.

    Uses libsonata directly and contains a bunch of workarounds to accomodate files
    created in the transition to SONATA. Also translates all GIDs from 0-based as on disk
    to the 1-based convention in Neurodamus.

    Will read each attribute for multiple GIDs at once and cache read data in a columnar
    fashion.

    FIXME Remove the caching at the np.recarray level.
    """

    SYNAPSE_INDEX_NAMES = ("synapse_index",)
    LOOKUP_BY_TARGET_IDS = True  # False to lookup by Source Ids
    # SynapseParameters knows how to load the parameters
    # Child classes of SonataReader can override this with another class
    # probably inherited from SynapseParameters to load a different set of parameters
    Parameters = SynapseParameters  # By default we load synapses
    EMPTY_DATA = {}

    custom_parameters = {"isec", "ipt", "offset"}
    """Custom parameters are skipped from direct loading and trigger _load_params_custom()"""

    parameter_mapping = {
        "weight": "conductance",
        "U": "u_syn",
        "D": "depression_time",
        "F": "facilitation_time",
        "DTC": "decay_time",
        "synType": "syn_type_id",
        "nrrp": "n_rrp_vesicles",
        "conductance_ratio": "conductance_scale_factor",
    }

    def __init__(self, edge_file, population=None, *_, **kw):
        self._ca_concentration = kw.get("extracellular_calcium")
        self._syn_params = {}  # Parameters cache by post-gid (previously loadedMap)
        self._open_file(edge_file, population, kw.get("verbose", False))
        # NOTE u_hill_coefficient and conductance_scale_factor are optional, BUT
        # while u_hill_coefficient can always be readif avail, conductance reader may not.
        self._uhill_property_avail = self.has_property("u_hill_coefficient")
        self._extra_fields = set()
        self._extra_scale_vars = []

    def configure_override(self, mod_override):
        if not mod_override:
            return

        override_helper = mod_override + "Helper"
        Nd.load_hoc(override_helper)

        # Read attribute names with format "attr1;attr2;attr3"
        attr_names = getattr(Nd, override_helper + "_NeededAttributes", None)
        if attr_names:
            log_verbose(
                'Reading parameters "{}" for mod override: {}'.format(
                    ", ".join(attr_names.split(";")), mod_override
                )
            )
            self._extra_fields = set(attr_names.split(";"))

        # Read attribute names with format "attr1;attr2;attr3"
        attr_names = getattr(Nd, override_helper + "_UHillScaleVariables", None)
        if attr_names:
            self._extra_scale_vars = attr_names.split(";")

    def get_synapse_parameters(self, gid) -> np.recarray:
        """Return the synapse parameters record array for the given gid,
        loading and caching it if needed.
        """
        syn_params = self._syn_params.get(gid)
        if syn_params is None:
            # prepare data for the gid
            data = self._data.get(gid)
            if data is None:  # not in _data
                self._preload_data_chunk([gid])
                data = self._data[gid]

            # create the synapse parameters array, already patched
            syn_params = self.Parameters.make_synapse_parameters_array(
                data, self._extra_fields, self._ca_concentration, self._extra_scale_vars
            )
            # cache the results
            self._syn_params[gid] = syn_params
        return syn_params

    def _open_file(self, src, population, _):
        """Initializes the reader, opens the synapse file"""
        try:
            from mpi4py import MPI

            hdf5_reader = libsonata.make_collective_reader(
                MPI.COMM_WORLD, collective_metadata=False, collective_transfer=True
            )
        except ModuleNotFoundError:
            hdf5_reader = libsonata.Hdf5Reader()

        storage = libsonata.EdgeStorage(src, hdf5_reader=hdf5_reader)
        if not population:
            assert len(storage.population_names) == 1, f"Populations: {storage.population_names}"
            population = next(iter(storage.population_names))
        self._population = storage.open_population(population)
        # A cache which stores all the fields for each gid. E.g. {1: {"sgid": property_numpy}}
        self._data = {}
        # A cache for connection counts, used mostly in dry run
        self._counts = {}

    def has_property(self, field_name):
        """Checks whether source data has the given additional field."""
        if field_name in self.SYNAPSE_INDEX_NAMES:
            return True
        return field_name in self._population.attribute_names

    def get_property(self, gid, field_name):
        """Retrieves a full pre-loaded property given a gid and the property name."""
        return self._data[gid][field_name]

    def preload_data(self, gids, minimal_mode=False):
        """Preload SONATA fields for the specified IDs.
        Set minimal_mode to True to read a single synapse per connection
        """
        # TODO: limit the number of cells per chunk in production.
        #       Ensuring the number of chunks must be the same in all ranks (collective)!
        CHUNK_SIZE = 1000
        if not minimal_mode or len(gids) < CHUNK_SIZE:
            return

        ranges = list(gen_ranges(len(gids), CHUNK_SIZE))
        for start, end in ProgressBar.iter(ranges, name="Prefetching"):
            self._preload_data_chunk(gids[start:end], minimal_mode)

    def _preload_data_chunk(self, gids, minimal_mode=False):  # noqa: C901
        """Preload all synapses for a number of gids, respecting Parameters and _extra_fields"""
        # NOTE: to disambiguate, gids are 1-based cell ids, while node_ids are 0-based sonata ids
        compute_fields = {"sgid", "tgid", *self.SYNAPSE_INDEX_NAMES}
        orig_needed_gids_set = set(gids) - set(self._data.keys())
        needed_gids = sorted(orig_needed_gids_set)

        def get_edge_and_lookup_gids(needed_gids: libsonata.Selection):
            """Retrieve edge and corresponding gid for"""
            node_ids = np.array(needed_gids, dtype="int64") - 1
            if self.LOOKUP_BY_TARGET_IDS:
                edge_ids = self._population.afferent_edges(node_ids)
                return edge_ids, self._population.target_nodes(edge_ids) + 1
            edge_ids = self._population.efferent_edges(node_ids)
            return edge_ids, self._population.source_nodes(edge_ids) + 1

        # NOTE: needed_edge_ids, lookup_gids are used in _populate and _read
        needed_edge_ids, lookup_gids = get_edge_and_lookup_gids(needed_gids)

        # Find and exclude gids without data
        different_gids_edge_i = np.diff(lookup_gids, prepend=np.nan).nonzero()[0]
        needed_gids = sorted(lookup_gids[different_gids_edge_i])
        for gid in orig_needed_gids_set - set(needed_gids):
            self._data.setdefault(gid, self.EMPTY_DATA)

        # In minimal mode read a single synapse (the first) of each target gid
        if minimal_mode:
            needed_edge_ids = libsonata.Selection(needed_edge_ids.flatten()[different_gids_edge_i])
            lookup_gids = lookup_gids[different_gids_edge_i]

        def _populate(field, data):
            # Populate cache. Unavailable entries are stored as a plain -1
            if data is None:
                data = -1
            for gid in needed_gids:
                existing_gid_data = self._data.setdefault(gid, {})
                existing_gid_data[field] = data if np.isscalar(data) else data[lookup_gids == gid]

        def _read(attribute, optional=False):
            if attribute in self._population.attribute_names:
                return self._population.get_attribute(attribute, needed_edge_ids)
            if optional:
                log_verbose("Defaulting to -1.0 for attribute %s", attribute)
                return -1
            raise AttributeError(f"Missing attribute {attribute} in the SONATA edge file")

        # Populate the opposite node id
        if self.LOOKUP_BY_TARGET_IDS:
            _populate("sgid", self._population.source_nodes(needed_edge_ids) + 1)
        else:
            _populate("tgid", self._population.target_nodes(needed_edge_ids) + 1)

        # Make synapse index in the file explicit
        for name in sorted(self.SYNAPSE_INDEX_NAMES):
            _populate(name, needed_edge_ids.flatten())

        # Generic synapse parameters
        fields_load_sonata = self.Parameters.fields(
            exclude=self.custom_parameters | compute_fields, with_translation=self.parameter_mapping
        )
        for field, sonata_attr, is_optional in sorted(fields_load_sonata):
            _populate(field, _read(sonata_attr, is_optional))

        if self.custom_parameters:
            if minimal_mode:
                _populate("isec", 0)
                _populate("ipt", -1)
                _populate("offset", 0)
                return  # done! Skip extra fields
            self._load_params_custom(_populate, _read)

        # Extend Gids data with the additional requested fields
        # This has to work for when we call preload() a second/third time
        # so we are unsure about which gids were loaded what properties
        # We nevertheless can skip any base fields
        extra_fields = self._extra_fields - (self.Parameters.all_fields() | compute_fields)
        for field in sorted(extra_fields):
            now_needed_gids = sorted(
                {
                    gid
                    for gid in gids
                    if (data := self._data[gid]) is not self.EMPTY_DATA and field not in data
                }
            )
            if needed_gids != now_needed_gids:
                needed_gids = now_needed_gids
                needed_edge_ids, lookup_gids = get_edge_and_lookup_gids(needed_gids)
            sonata_attr = self.parameter_mapping.get(field, field)
            _populate(field, _read(sonata_attr))

    def _load_params_custom(self, _populate, _read):
        # Position of the synapse
        if self.has_property("afferent_section_id"):
            _populate("isec", _read("afferent_section_id"))
            # SONATA compliant synapse position: (section, section_fraction) takes precedence
            # over the older (section, segment, segment_offset) synapse position.
            #
            # Re-using field names for historical reason, where -1 means N/A.
            # FIXME Use dedicated fields
            if self.has_property("afferent_section_pos"):
                _populate("ipt", -1)
                _populate("offset", _read("afferent_section_pos"))
            # This was a temporary naming scheme
            # FIXME Circuits using this field should be fixed
            elif self.has_property("afferent_section_fraction"):
                logging.warning(
                    "Circuit uses non-standard compliant attribute `afferent_section_fraction`"
                )
                _populate("ipt", -1)
                _populate("offset", _read("afferent_section_fraction"))
            else:
                logging.warning(
                    "Circuit is missing standard compliant attribute `afferent_section_pos`"
                )
                _populate("ipt", _read("afferent_segment_id"))
                _populate("offset", _read("afferent_segment_offset"))
        else:
            # FIXME All this should go the way of the dodo
            logging.warning(
                "Circuit uses attribute notation using `morpho_` and is not SONATA compliant"
            )
            _populate("isec", _read("morpho_section_id_post"))
            if self.has_property("morpho_section_fraction_post"):
                _populate("ipt", -1)
                _populate("offset", _read("morpho_section_fraction_post"))
            else:
                _populate("ipt", _read("morpho_segment_id_post"))
                _populate("offset", _read("morpho_offset_segment_post"))

    def get_counts(self, tgids):
        """Counts synapses for the given target neuron ids. Returns a dict"""
        node_ids = tgids - 1
        edge_ids = self._population.afferent_edges(node_ids)
        target_nodes = self._population.target_nodes(edge_ids)
        unique_nodes, counts = np.unique(target_nodes, return_counts=True)
        unique_gids = unique_nodes + 1
        counts_dict = dict(zip(unique_gids, counts))
        for gid in tgids:
            counts_dict.setdefault(gid, 0)
        return counts_dict

    def get_conn_counts(self, tgids):
        """Counts synapses per connetion for all the given target neuron ids.
        Returns a dict whose value is a numpy stuctured array
        """
        if missing_gids := set(tgids) - set(self._counts):
            missing_gids = np.fromiter(missing_gids, dtype="uint32")
            missing_gids.sort()
            missing_nodes = missing_gids - 1
            edge_ids = self._population.afferent_edges(missing_nodes)
            target_nodes = self._population.target_nodes(edge_ids)
            source_nodes = self._population.source_nodes(edge_ids)
            connections = np.empty(len(target_nodes), dtype="uint64,uint64")
            connections["f0"] = target_nodes + 1  # nodes to 1-based gids
            connections["f1"] = source_nodes + 1

            tgt_src_pairs, counts = np.unique(connections, return_counts=True)
            pairs_start_i = np.diff(tgt_src_pairs["f0"], prepend=np.nan, append=np.nan).nonzero()[0]

            for conn_i, start_i in enumerate(pairs_start_i[:-1]):
                end_i = pairs_start_i[conn_i + 1]
                tgid = tgt_src_pairs["f0"][start_i]
                tgid_counts = {tgt_src_pairs["f1"][j]: counts[j] for j in range(start_i, end_i)}
                self._counts[tgid] = tgid_counts

        return {tgid: self._counts.get(tgid, self.EMPTY_DATA) for tgid in tgids}
