"""Function for dumping cell state from NEURON context"""

import json
import logging

FILTER_SYNAPSE_ATTRS = ["delay_times", "delay_weights", "rng"]


def dump_cellstate(pc, cvode, gid, outputfile="cellstate.json"):
    """Dump cell, synapse, netcon states from NEURON context
    Args:
        pc: NEURON parallel contect
        cvode: NEURON CVode context, to get netcons list
        gid: cell gid in NEURON context
    """
    logging.info("Dump cell state for id %s", gid)
    cell = pc.gid2cell(gid)
    name = cell.hname()
    # remove the cell index from names
    cell_name = name[: name.find("[")]
    cell_prefix = name + "."
    res = {cell_name: {"gid": gid}}
    res[cell_name].update(dump_cell(cell))
    res[cell_name]["synapses"] = dump_synapses(cell.synlist, remove_prefix=cell_prefix)
    nclist = cvode.netconlist("", cell, "")
    res[cell_name]["n_netcons"] = nclist.count()
    res[cell_name]["netcons"] = dump_netcons(nclist, remove_prefix=cell_prefix)

    with open(outputfile, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


def dump_cell(cell) -> dict:
    """Dump cell state from NEURON context
    Args:
        cell: NEURON cell object
    """
    res = _read_object_attrs(cell)
    res["sections"] = []
    cell_name = cell.hname()
    for sec in cell.all:
        res_sec = {}
        sec_name = sec.hname()
        sec_name = sec_name.replace(cell_name + ".", "")
        res_sec["name"] = sec_name
        res_sec.update(_read_object_attrs(sec))
        res_sec["segments"] = []
        for seg in sec:
            attrs = {"x": seg.x}
            attrs.update(_read_object_attrs(seg))
            for key, item in attrs.items():
                # item is likely an nrn.Mechanism object
                if not isinstance(item, (int, float, str, list, dict, set)):
                    vals = _read_object_attrs(item)
                    attrs[key] = vals
            res_sec["segments"].append(attrs)
        res["sections"].append(res_sec)

    res["n_synapses"] = cell.synlist.count()
    return res


def dump_synapses(synlist, remove_prefix) -> list:
    """Dump states of synapses
    Args:
        synlist: Hoc list of synapse objects
        remove_prefix: remove prefix from key name
    """
    res = []
    for syn in synlist:
        attrs = {"name": syn.hname()}
        attrs.update(_read_object_attrs(syn, FILTER_SYNAPSE_ATTRS))
        attrs["location"] = syn.get_loc()
        attrs["segment"] = str(syn.get_segment()).removeprefix(remove_prefix)
        res.append(attrs)
    return res


def dump_netcons(nclist, remove_prefix) -> list:
    """Dump states of netcons from NEURON CVode
    Args:
        nclist: NEURON netcons list
        remove_prefix: remove prefix from key name
    """
    res = []
    for nc in nclist:
        attrs = {"name": nc.hname()}
        tsyn = nc.syn()
        attrs["target_syn"] = str(tsyn).removeprefix(remove_prefix)
        attrs["target_syn_segment"] = str(tsyn.get_segment()).removeprefix(remove_prefix)
        attrs["afferent_section_position"] = tsyn.get_segment().x
        attrs["srcgid"] = nc.srcgid()
        attrs["active"] = nc.active()
        attrs["weight"] = nc.weight[0]
        attrs.update(_read_object_attrs(nc))
        res.append(attrs)
    return res


def _read_object_attrs(obj, filter_keys=None):
    res = {}
    for x in dir(obj):
        if (
            (not filter_keys or x not in filter_keys)
            and not x.startswith("__")
            and not callable(getattr(obj, x))
        ):
            attr = getattr(obj, x)
            if isinstance(attr, float):
                attr = round(attr, 14)
            res[x] = attr
    return res
