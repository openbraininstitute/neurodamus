def dump_cellstate(cell) -> dict:
    res = _read_object_attrs(cell)
    res["n_sections"] = -1
    res["sections"] = []
    cell_name = cell.hname()
    nsec = 0
    for sec in cell.all:
        nsec += 1
        res_sec = {}
        sec_name = sec.hname()
        sec_name = sec_name.replace(cell_name + ".", "")
        res_sec["name"] = sec_name
        res_sec.update(_read_object_attrs(sec))
        res_sec["segments"] = []
        # res_sec[sec.hname()]["segments"] = []
        for seg in sec:
            attrs = {"x": seg.x, 
                     "node_index": seg.node_index()
                     }
            # attrs['name'] = str(seg)
            attrs.update(_read_object_attrs(seg))
            for key, item in attrs.items():
                # item is likely an nrn.Mechanism object
                if not isinstance(item, (int, float, str, list, dict, set)):
                    vals = _read_object_attrs(item)
                    attrs[key] = vals
            res_sec["segments"].append(attrs)
        res["n_sections"] = nsec
        res["sections"].append(res_sec)

    res["n_synapses"] = cell.synlist.count()
    res["synapses"] = []
    for syn in cell.synlist:
        attrs = {"name": syn.hname()}
        attrs.update(_read_object_attrs(syn))
        attrs["location"] = syn.get_loc()
        attrs["segment"] = str(syn.get_segment())
        res["synapses"].append(attrs)
    return res


def dump_nclist(nclist) -> list:
    res = []
    for nc in nclist:
        attrs = {"name": nc.hname()}
        if nc.precell():
            attrs["precell"] = nc.precell().hname()
        elif nc.pre():
            attrs["pre"] = nc.pre().hname()
        attrs["srcgid"] = nc.srcgid()
        attrs["active"] = nc.active()
        attrs["weight"] = nc.weight[0]
        attrs.update(_read_object_attrs(nc))
        res.append(attrs)
    return res


def _read_object_attrs(obj):
    return {x: getattr(obj, x) for x in dir(obj)
            if not x.startswith("__") and not callable(getattr(obj, x))}
