import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from libsonata import EdgeStorage

from neurodamus.core.configuration import SimConfig


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
            raise TypeError(
                f"Field type missmatch for the values of key {k}: "
                f"{parent[k]} ({type(parent[k])}) != {child[k]} ({type(child[k])})"
            )
        if isinstance(parent[k], dict):
            return merge_dicts(parent[k], child[k])
        return child[k]

    return {k: merge_vals(k, parent, child) for k in set(parent) | set(child)}


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


def check_directory(dir_name: Path):
    """ Check if a directory exists and is not empty """
    assert dir_name.is_dir(), f"{str(dir_name)} doesn't exist"
    assert any(dir_name.iterdir()), f"{str(dir_name)} is empty"


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
    assert decay_time == _get_attr("decay_time", kwargs, edges, selection, syn_id)
    assert syn.Use == _get_attr("u_syn", kwargs, edges, selection, syn_id)
    assert syn.Dep == _get_attr("depression_time", kwargs, edges, selection, syn_id)
    assert syn.Fac == _get_attr("facilitation_time", kwargs, edges, selection, syn_id)

    if _get_attr("n_rrp_vesicles", kwargs, edges, selection, syn_id) >= 0:
        assert syn.Nrrp == _get_attr("n_rrp_vesicles", kwargs, edges, selection, syn_id)


def check_signal_peaks(x, ref_peaks_pos, threshold=1):
    """
    Check the given signal peaks comparing with the given
    reference

    Args:
        x: given signal, typically voltage.
        ref_peaks_pos: the position of the signal peaks
        taken as reference.
        threshold: peak detection threshold measured with
        respect of the surrounding baseline of the signal

    Raises:
        AssertionError: If any of the reference peak
        positions doesn't match with the obtained peaks
    """
    peaks_pos = find_peaks(x, prominence=threshold)[0]
    np.testing.assert_equal(peaks_pos, ref_peaks_pos)
