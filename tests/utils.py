import json
from pathlib import Path

import numpy as np
from libsonata import EdgeStorage, SpikeReader
from scipy.signal import find_peaks
from collections import defaultdict
from collections.abc import Iterable

from neurodamus.core import NeuronWrapper as Nd
from neurodamus.core.configuration import SimConfig
from neurodamus.target_manager import TargetManager, TargetSpec


def merge_dicts(parent: dict, child: dict):
    """Merge dictionaries recursively (in case of nested dicts) giving priority to child over parent
    for ties. Values of matching keys must match or a TypeError is raised.

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
        if k not in parent:
            return child[k]
        if k not in child:
            return parent[k]
        if type(parent[k]) is not type(child[k]):
            if not isinstance(parent[k], (int, float)) or not isinstance(child[k], (int, float)):
                raise TypeError(
                    f"Field type missmatch for the values of key {k}: "
                    f"{parent[k]} ({type(parent[k])}) != {child[k]} ({type(child[k])})"
                )
        if isinstance(parent[k], dict):
            return merge_dicts(parent[k], child[k])
        return child[k]

    return {k: merge_vals(k, parent, child) for k in set(parent) | set(child)}


def defaultdict_to_standard_types(obj):
    """Recursively converts defaultdicts with iterable values to standard Python types."""
    if isinstance(obj, defaultdict) or isinstance(obj, dict):
        return {key: defaultdict_to_standard_types(value) for key, value in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return list(obj)
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
    selection = edges.afferent_edges(tgt_rawgid - 1)
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


def check_signal_peaks(x, ref_peaks_pos, threshold=1, tolerance=0):
    """
    Check the given signal peaks comparing with the given
    reference

    Args:
        x: given signal, typically voltage.
        ref_peaks_pos: the position of the signal peaks
        taken as reference.
        threshold: peak detection threshold measured with
        respect of the surrounding baseline of the signal
        tolerance: peak detection tolerance window per peak

    Raises:
        AssertionError: If any of the reference peak
        positions doesn't match with the obtained peaks
    """
    peaks_pos = find_peaks(x, prominence=threshold)[0]
    np.testing.assert_allclose(peaks_pos, ref_peaks_pos, atol=tolerance)


def record_compartment_report(rep_conf: dict, target_manager: TargetManager):
    """For compartment report, retrieve segments, and record the pointer of reporting variable
    More details in NEURON Vector.record()
    """
    rep_type = rep_conf["Type"]
    assert rep_type == "compartment", "Report type is not supported"
    sections = rep_conf.get("Sections")
    compartments = rep_conf.get("Compartments")
    variable_name = rep_conf["ReportOn"]
    start_time = rep_conf["StartTime"]
    stop_time = rep_conf["EndTime"]
    dt = rep_conf["Dt"]

    tvec = Nd.Vector()
    tvec.indgen(start_time, stop_time, dt)

    target_spec = TargetSpec(rep_conf["Target"])
    target = target_manager.get_target(target_spec)
    sum_currents_into_soma = sections == "soma" and compartments == "center"
    # In case of summation in the soma, we need all points anyway
    if sum_currents_into_soma and rep_type == "Summation":
        sections = "all"
        compartments = "all"
    points = target_manager.get_point_list(target, sections=sections, compartments=compartments)
    recorder = []
    for point in points:
        gid = point.gid
        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            # Enable fast_imem calculation in Neuron
            if variable_name == "i_membrane":
                Nd.cvode.use_fast_imem(1)
                variable_name = "i_membrane_"
            var_ref = getattr(section(x), "_ref_" + variable_name)
            voltage_vec = Nd.Vector()
            voltage_vec.record(var_ref, tvec)
            segname = str(section(x))
            segname = segname[segname.find(".") + 1:]
            recorder.append((gid, segname, voltage_vec))
    return recorder, tvec


def write_ascii_report(filename, recorder, tvec):
    """Write out the report in ASCII format"""
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
