def dump_cellstate(cell):
    res = {}
    res[cell.hname()] = _read_object_attrs(cell)
    nsec = 0
    for sec in cell.all:
        nsec += 1
        res[sec.hname()] = _read_object_attrs(sec)
        res[sec.hname()]["segments"] = {}
        for seg in sec:
            attrs = _read_object_attrs(seg)
            for key, item in attrs.items():
                # item is likely an nrn.Mechanism object
                if not isinstance(item, (int, float, str, list, dict, set)):
                    vals = _read_object_attrs(item)
                    attrs[key] = vals
            attrs["node_index"] = seg.node_index()
            res[sec.hname()]["segments"][str(seg)] = attrs
    res[cell.hname()]["nsec"] = nsec
    res_syns = {}
    res_syns["count"] = cell.synlist.count()
    for syn in cell.synlist:
        res_syns[syn.hname()] = _read_object_attrs(syn)
        res_syns[syn.hname()]["location"] = syn.get_loc()
        res_syns[syn.hname()]["segment"] = str(syn.get_segment())
    res["synapses"] = res_syns

    return res


def _read_object_attrs(obj):
    return {x: getattr(obj, x) for x in dir(obj)
            if not x.startswith("__") and not callable(getattr(obj, x))}
