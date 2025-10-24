import json
import h5py
from pathlib import Path
import logging

import numpy as np
from libsonata import EdgeStorage, SpikeReader, ElementReportReader
from scipy.signal import find_peaks
from collections import defaultdict
from collections.abc import Iterable

from neurodamus.core import NeuronWrapper as Nd
from neurodamus.core.configuration import SimConfig
from neurodamus.target_manager import TargetManager, TargetSpec
from neurodamus.report import Report
from neurodamus.report_parameters import create_report_parameters, CompartmentType, ReportType, SectionType

from typing import List, Dict, Tuple
import pandas as pd
import copy


def merge_dicts(parent: dict, child: dict):
    """Merge dictionaries recursively (in case of nested dicts) giving priority to child over parent
    for ties. Values of matching keys must match or a TypeError is raised.

    Special values/keys:
    - If a key in `child` has value "delete_field", it will be removed from the result.
    - If a dictionary (nested or not) in `child` contains the key "override_field", it replaces the corresponding 
      `parent` sub-dictionary entirely (ignoring merging).

    Imported from MultiscaleRun.

    Args:
        parent: parent dict
        child: child dict (priority)

    Returns:
        dict: merged dict following the rules listed before

    Example::

        >>> parent = {"A":1, "B":{"a":1, "b":2}, "C": 2}
        >>> child = {"A":2, "B":{"a":2, "c":3}, "D": 3}
        >>> merge_dicts(parent, child)
        {"A":2, "B":{"a":2, "b":2, "c":3}, "C": 2, "D": 3}
    """

    def sanitize_dict(d):
        if isinstance(d, dict):
            # Delete the key if present
            d.pop("override_field", None)
            for key, value in d.items():
                sanitize_dict(value)
        elif isinstance(d, list):
            for item in d:
                sanitize_dict(item)

    def merge_vals(k, parent: dict, child: dict):
        """Merging logic.

        Args:
            k (key type): the key can be in either parent, child or both.
            parent: parent dict.
            child: child dict (priority).

        Raises:
            TypeError: in case the key is present in both parent and child and the type missmatches.

        Returns:
            value type: merged version of the values possibly found in child and/or parent.
        """

        if k not in child:
            return parent[k]
        if k not in parent:
            return child[k]
        if type(parent[k]) is not type(child[k]):
            if not isinstance(parent[k], (int, float)) or not isinstance(child[k], (int, float)):
                raise TypeError(
                    f"Field type missmatch for the values of key {k}: "
                    f"{parent[k]} ({type(parent[k])}) != {child[k]} ({type(child[k])})"
                )
        if isinstance(parent[k], dict):
            if "override_field" in child[k]:
                return child[k]

            return merge_dicts(parent[k], child[k])
        return child[k]
    
    ans = {
        k: merge_vals(k, parent, child)
        for k in set(parent) | set(child)
        if not isinstance(child, dict) or k not in child or child[k] != "delete_field"
    } if "override_field" not in child else copy.deepcopy(child)
    sanitize_dict(ans)
    return ans

def defaultdict_to_standard_types(obj):
    """Recursively converts defaultdicts with iterable values to standard Python types."""
    if isinstance(obj, (defaultdict, dict)):
        return {key: defaultdict_to_standard_types(value) for key, value in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return [defaultdict_to_standard_types(x) for x in obj]
    elif isinstance(obj, np.generic):  # convert NumPy scalars to Python scalars
        return obj.item()
    return obj

def check_is_subset(dic, subset):
    """Checks if subset is a subset of the original dict"""
    try:
        merged = merge_dicts(dic, subset)
    except TypeError:
        assert False
    assert dic == merged


def get_edge_data(nd, src_pop: str, src_rawgid: int, tgt_pop: str, tgt_rawgid: int):
    """ Convenience function to retrieve gids, edges, and selection.

    Nd is neurodamus.code.Neurodamus
    """
    tgt_pop_offset = nd.circuits.get_node_manager(tgt_pop).local_nodes.offset
    tgt_gid = tgt_rawgid + tgt_pop_offset
    src_pop_offset = nd.circuits.get_node_manager(src_pop).local_nodes.offset
    src_gid = src_rawgid + src_pop_offset

    if src_pop == tgt_pop:
        edges_file, edge_pop = \
            nd.circuits.get_edge_managers(tgt_pop, tgt_pop)[0].circuit_conf.nrnPath.split(":")
    else:
        edges_file, edge_pop = \
            nd.circuits.get_edge_managers(src_pop, tgt_pop)[0].circuit_conf["Path"].split(":")
    edge_storage = EdgeStorage(edges_file)
    edges = edge_storage.open_population(edge_pop)
    selection = edges.afferent_edges(tgt_rawgid)
    return src_gid, tgt_gid, edges, selection


def _get_attr(name, kwargs, edges, selection, syn_id):
    """
    Retrieve an attribute value from `kwargs` if present, otherwise
    from `edges`.
    """
    return kwargs.get(name, edges.get_attribute(name, selection)[syn_id])


def compare_json_files(res_file: Path, ref_file: Path):
    """
    Compare two JSON files to check if their contents are identical.

    This function opens and loads two JSON files, `res_file` and `ref_file`,
    and compares their contents. If the files are identical, the function
    completes without any errors. If they differ, an assertion error will
    be raised.

    Args:
        res_file (Path): The path to the result JSON file to compare.
        ref_file (Path): The path to the reference JSON file to compare against.

    Raises:
        AssertionError: If the two JSON files have different contents.
        FileNotFoundError: If either of the provided file paths does not exist.
    """
    assert res_file.exists()
    assert ref_file.exists()
    with open(res_file) as f_res:
        result = json.load(f_res)
    with open(ref_file) as f_ref:
        reference = json.load(f_ref)
    assert result == reference


def check_directory(dir_name: "Path | str"):
    """ Check if a directory exists and is not empty """
    dir_name = Path(dir_name)
    assert dir_name.is_dir(), f"{dir_name} doesn't exist"
    assert any(dir_name.iterdir()), f"{dir_name} is empty"


def check_netcons(ref_srcgid, nclist, edges, selection, **kwargs):
    """
    Convenience function to validate all netcons in `nclist`
    by using the underlying `check_netcon` for each item.
    """
    assert len(nclist) == selection.flat_size
    # if the list is empty, just check that
    assert nclist.count()
    for nc_id, nc in enumerate(nclist):
        check_netcon(ref_srcgid, nc_id, nc, edges, selection, **kwargs)


def check_netcon(src_gid, nc_id, nc, edges, selection, **kwargs):
    """
    Validate the attributes of a single netcon against expected values.

    This function checks the following attributes of a given netcon:
    - `src_gid`: Ensures it matches the reference source ID (`rsc_gid`).
    - `weight`: Connection overrides uses this value to scale the
    `conductance` in the edge files.
    - `delay`: Compares it with the expected delay value, allowing for
    a relative tolerance.
    - `threshold`: Compares it with the `spike_threshold` value from `kwargs`
    or the default configuration.
    - `x`: Compares it with the initial voltage (`v_init`), again using `kwargs`
    or the default configuration.

    `kwargs` allows for overriding the default or edge-based values
    for the above attributes.

    Args:
        ref_srcgid (int): The reference source ID to be compared against `nc.srcgid()`.
        nc_id (int): The index of the current netcon in the list, used to
        fetch specific attributes from edges.
        nc: The netcon object to be validated.
        edges: An object representing synaptic edges, used to retrieve attribute values.
        selection: The selection criterion used for fetching attributes for
        the given netcon.
        **kwargs: Optional arguments to override default values
        for conductance, delay, spike threshold, and initial voltage.

    Raises:
        AssertionError: If any of the attribute checks fail.
    """

    assert nc.srcgid() == src_gid
    assert np.isclose(
        nc.weight[0],
        kwargs.get(
            "weight",
            1.0) *
        edges.get_attribute(
            "conductance",
            selection)[nc_id])
    assert np.isclose(nc.delay, _get_attr("delay", kwargs, edges, selection, nc_id))
    assert np.isclose(nc.threshold, kwargs.get("spike_threshold", SimConfig.spike_threshold))
    assert np.isclose(nc.x, kwargs.get("v_init", SimConfig.v_init))


def check_synapses(nclist, edges, selection, **kwargs):
    """
    Convenience function to validate all synapses in `nclist`
    by using the underlying `check_synapse` for each item.
    """
    assert len(nclist) == selection.flat_size
    # if the list is empty, just check that
    assert nclist.count()
    for nc in nclist:
        check_synapse(nc.syn(), edges, selection, **kwargs)


def check_synapse(syn, edges, selection, **kwargs):
    """
    Check the state of a synapse from the NEURON model in comparison to the
    libsonata reader.

    This function verifies that the attributes of the synapse match the expected values
    based on its type and the provided parameters. If `kwargs` are provided, they will
    override the corresponding values from the `edges` object.

    Args:
        syn: The synapse object to check, which is expected to have attributes like
        `synapseID`, `hname()`,
             `tau_d_GABAA`, `tau_d_AMPA`, `Use`, `Dep`, `Fac`, `Nrrp`.
        edges: An object representing the synaptic edges, from which various attributes are fetched
               using the `get_attribute` method.
        selection: The selection criterion for retrieving attributes from `edges`.
        kwargs: Optional key-value pairs that can override the default attribute
        values from `edges`.
                Keys include `decay_time`, `u_syn`, `depression_time`,
                `facilitation_time`, and `n_rrp_vesicles`.

    Raises:
        AssertionError: If any of the synaptic attributes do not match the expected values.
    """

    expected_types = ["ProbGABAAB_EMS", "ProbAMPANMDA_EMS"]

    syn_id = int(syn.synapseID)

    syn_type = kwargs.get("hname", syn.hname()).split("[")[0]
    # for now we test only with the basic ones
    assert syn_type in expected_types

    if "hname" not in kwargs:
        syn_type_id = edges.get_attribute("syn_type_id", selection)[syn_id]
        exp_type = expected_types[0] if syn_type_id < 100 else expected_types[1]
        assert syn_type == exp_type, f"{syn_type}"

    decay_time = syn.tau_d_GABAA if syn_type == "ProbGABAAB_EMS" else syn.tau_d_AMPA
    assert np.isclose(decay_time, _get_attr("decay_time", kwargs, edges, selection, syn_id))
    assert np.isclose(syn.Use, _get_attr("u_syn", kwargs, edges, selection, syn_id))
    assert np.isclose(syn.Dep, _get_attr("depression_time", kwargs, edges, selection, syn_id))
    assert np.isclose(syn.Fac, _get_attr("facilitation_time", kwargs, edges, selection, syn_id))

    if _get_attr("n_rrp_vesicles", kwargs, edges, selection, syn_id) >= 0:
        assert np.isclose(syn.Nrrp, _get_attr("n_rrp_vesicles", kwargs, edges, selection, syn_id))
    assert syn.Nrrp > 0
    assert int(syn.Nrrp) == syn.Nrrp

    # check NMDA ratio if existing
    if "NMDA_ratio" in kwargs and hasattr(syn, "NMDA_ratio"):
        assert np.isclose(syn.NMDA_ratio, kwargs["NMDA_ratio"])

def record_compartment_reports(target_manager: TargetManager, nd_t=0):
    """For compartment report, retrieve segments, and record the pointer of reporting variable
    More details in NEURON Vector.record()

    This avoids libsonatareport. Additional tests with libsonatareport in integration-e2e
    """
    ascii_recorders = {}


    reports_conf = {name: conf for name, conf in SimConfig.reports.items() if conf["Enabled"]}

    for rep_name, rep_conf in reports_conf.items():
        target_spec = TargetSpec(rep_conf["Target"])
        target = target_manager.get_target(target_spec)

        rep_params = create_report_parameters(sim_end=SimConfig.run_conf["Duration"], nd_t=nd_t, output_root=SimConfig.output_root, rep_name=rep_name, rep_conf=rep_conf, target=target, buffer_size=8)

        if rep_params.type != ReportType.COMPARTMENT:
            continue

        tvec = Nd.Vector()
        tvec.indgen(rep_params.start, rep_params.end, rep_params.dt)

        sections, compartments = rep_params.sections, rep_params.compartments
        if rep_params.type == ReportType.SUMMATION and sections == SectionType.SOMA:
            sections, compartments = SectionType.ALL, CompartmentType.ALL
        points = rep_params.target.get_point_list(
            cell_manager=target_manager._cell_manager, section_type=sections, compartment_type=compartments
        )

        recorder = []

        variables = Report.parse_variable_names(rep_params.report_on)
        assert len(variables) == 1
        mechanism, variable_name = variables[0]
        
        for point in points:
            gid = point.gid
            for i, sc in enumerate(point.sclst):
                section = sc.sec
                x = point.x[i]

                var_refs = Report.get_var_refs(section, x, mechanism, variable_name)
                assert len(var_refs) == 1
                trace = Nd.Vector()
                trace.record(var_refs[0], tvec)
                segname = str(section(x))
                segname = segname[segname.find(".") + 1:]
                recorder.append((gid, segname, trace))

        ascii_recorders[rep_name] = (recorder, tvec)
    return ascii_recorders

def write_ascii_reports(ascii_recorders, output_path):
    """Write out the report in ASCII format"""
    for rep_name, (recorder, tvec) in ascii_recorders.items():
        filename = Path(output_path) / (rep_name + ".txt")
        with open(filename, "w") as f:
            f.write(f"{'cell_id':<10}{'seg_name':<20}{'time':<20}{'data':<20}\n")
            for gid, secname, data_vec in recorder:
                f.writelines(f"{gid:<10}{secname:<20}{t:<20.4f}{data:<20.4f}\n"
                            for t, data in zip(tvec, data_vec))


def read_ascii_report(filename):
    """Read an ASCII report and return report data in format:(gid, seg_name, time, report_variable)
    """
    data_vec = []
    with open(filename) as f:
        next(f)  # skip header
        for line in f:
            gid, seg_name, time, data = line.split()
            data_vec.append((int(gid), seg_name, float(time), float(data)))
    return data_vec


def read_sonata_spike_file(spike_file):
    """ Read a spike file and return the traces """
    spikes = SpikeReader(spike_file)
    pop_name = spikes.get_population_names()[0]
    data = spikes[pop_name].get()
    timestamps = np.array([x[1] for x in data])
    spike_gids = np.array([x[0] for x in data])
    return timestamps, spike_gids


def compare_outdat_files(file1, file2, start_time=None, end_time=None):
    """Compare two event files within an optional time frame."""
    start = start_time if start_time is not None else -np.inf
    end = end_time if end_time is not None else np.inf

    def load_and_filter(file_path):
        events = np.loadtxt(file_path)
        return events[(events[:, 0] >= start) & (events[:, 0] <= end)]

    events1 = load_and_filter(file1)
    events2 = load_and_filter(file2)

    return np.array_equal(np.sort(events1, axis=0), np.sort(events2, axis=0))

class ReportReader:
    def __init__(self, file: str):
        self._reader = ElementReportReader(file)
        self.populations: Dict[str, Tuple[List[int], pd.DataFrame]] = {}

        for name in sorted(self._reader.get_population_names()):
            pop = self._reader[name]
            node_ids = sorted(pop.get_node_ids())
            data = pop.get()

            # stable sort for ties
            df = pd.DataFrame(
                data.data,
                columns=pd.MultiIndex.from_arrays(data.ids.T),
                index=data.times
            ).sort_index(axis=1)

            self.populations[name] = (node_ids, df)

    def __eq__(self, other: object) -> bool:
        return self.allclose(other)

    def allclose(self, other: object, rtol=1e-16, atol=1e-16) -> bool:
        """
        Compare two ReportReader instances for approximate equality.
        
        Rtol and atol are the relative and absolute tolerances. The default values are
        the standards for numpy.allclose."""
        if not isinstance(other, ReportReader):
            return NotImplemented

        if set(self.populations.keys()) != set(other.populations.keys()):
            return False

        for name in self.populations:
            nodes1, df1 = self.populations[name]
            nodes2, df2 = other.populations[name]

            if nodes1 != nodes2:
                return False

            # coreneuron has sometimes garbage for the first line
            # erro thresholds as for old bb5 itegration report tests
            if not np.allclose(df1.values[1:], df2.values[1:], rtol=rtol, atol=atol):
                return False

        return True
    
    def convert_to_summation(self) -> None:
        new_populations = {}

        for name, (nodes, df) in self.populations.items():
            if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 1:
                new_df = df.groupby(level=0, axis=1).sum()
                # force 2-level MultiIndex with second level zeros
                new_df.columns = pd.MultiIndex.from_arrays([
                    new_df.columns,
                    [0] * len(new_df.columns)
                ])
                new_nodes = sorted(new_df.columns.get_level_values(0).tolist())
            else:
                new_df = df.copy()
                new_nodes = nodes

            new_populations[name] = (new_nodes, new_df)

        self.populations = new_populations

    def reduce_to_compartment_set_report(self, population: str, positions: List[int]) -> None:
        """
        Create and return a new ReportReader reduced to the selected columns
        of a population’s compartment set report. The original object is not
        modified
        """
        if population not in self.populations:
            raise ValueError(f"Population '{population}' not found in report.")

        nodes, df = self.populations[population]

        if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
            raise ValueError("Expected columns to be a 2-level MultiIndex.")

        # Use .iloc to select columns by position, preserving duplicates and order
        new_df = df.iloc[:, positions]

        # No need to rebuild MultiIndex — iloc preserves it
        # Extract node IDs from level 0 (with repetitions, in order)
        new_nodes = sorted(list(set([col[0] for col in new_df.columns])))

        new_reader = ReportReader.__new__(ReportReader)  # bypass __init__
        new_reader._reader = self._reader
        new_reader.populations = {population: (new_nodes, new_df)}

        return new_reader

    def __repr__(self) -> str:
        lines = [f"ReportReader with {len(self.populations)} populations:"]
        for name, (nodes, df) in self.populations.items():
            nodes_str = ", ".join(str(n) for n in nodes)
            lines.append(f"  - {name}: {len(nodes)} nodes, shape={df.shape}")
            lines.append(f"      node_ids: [{nodes_str}]")

            columns = df.columns
            cols_str = ", ".join(f"({a},{b})" for a, b in columns)

            lines.append(f"      columns: [{cols_str}]")

        return "\n".join(lines)

    def __add__(self, other: object) -> "ReportReader":
        """
        Add two ReportReader instances by element-wise summing their population data.
        """

        if not isinstance(other, ReportReader):
            return NotImplemented

        if set(self.populations.keys()) != set(other.populations.keys()):
            raise ValueError("ReportReaders have different populations.")

        new_populations = {}

        for name in self.populations:
            nodes1, df1 = self.populations[name]
            nodes2, df2 = other.populations[name]

            if nodes1 != nodes2:
                raise ValueError(f"Node IDs differ for population '{name}'.")

            if not df1.columns.equals(df2.columns):
                raise ValueError(f"DataFrame columns differ for population '{name}'.")

            if not df1.index.equals(df2.index):
                raise ValueError(f"DataFrame indices differ for population '{name}'.")

            new_df = df1 + df2  # element-wise addition

            new_populations[name] = (nodes1, new_df)

        # Create a new ReportReader instance without re-reading file
        new_report = ReportReader.__new__(ReportReader)
        new_report.populations = new_populations
        new_report._reader = None  # or keep from self if needed

        return new_report


