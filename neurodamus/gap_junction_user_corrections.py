# @file gap_junction_user_corrections.py
# @brief Script for loading user corrections on gap junction connectivity
# @author Oren Amsalem, Weina Ji
# @date 2024-09-09


import logging

from .core import MPI, NeuronWrapper as Nd
from .core.configuration import ConfigurationError, SimConfig

non_stochastic_mechs = [
    "NaTs2_t",
    "SKv3_1",
    "Nap_Et2",
    "Ih",
    "Im",
    "KdShu2007",
    "K_Pst",
    "K_Tst",
    "Ca",
    "SK_E2",
    "Ca_LVAst",
    "CaDynamics_E2",
    "NaTa_t",
    "CaDynamics_DC0",
    "Ca_HVA2",
    "NaTg",
    "TC_cad",
    "TC_ih_Bud97",
    "TC_Nap_Et2",
    "TC_iA",
    "TC_iL",
    "TC_HH",
    "TC_iT_Des98",
    "kdrb",
    "na3",
    "kap",
    "hd",
    "can",
    "cal",
    "cat",
    "cagk",
    "kca",
    "cacum",
    "kdb",
    "kmb",
    "kad",
    "nax",
    "cacumb",
]

stochastic_mechs = ["StochKv", "StochKv2", "StochKv3"]


def load_user_modifications(gj_manager):  # noqa: C901
    """Apply user modifications on gap junction connections, designed by @Oren Amsalem
    The modification parameters should be in the "beta_features" section of
    the simulation config file.
    """
    node_manager = gj_manager.cell_manager
    settings = SimConfig.beta_features
    gjc = settings.get("gjc")

    # deterministic_StochKv
    if settings.get("deterministic_stoch"):
        logging.info("Set deterministic = 1 for StochKv")
        _deterministic_stoch(node_manager)

    # update gap conductance
    if settings.get("procedure_type") in {"validation_sim", "find_holding_current"}:
        process_gap_conns = _update_conductance(gjc, gj_manager)
        all_ranks_total = int(MPI.py_sum(process_gap_conns, 0))
        logging.info("Set GJc = %s for %s gap synapses", gjc, all_ranks_total)

    # remove active channels
    remove_channels = settings.get("remove_channels")
    if remove_channels:
        if remove_channels == "all":
            rm_mechanisms = non_stochastic_mechs + stochastic_mechs
        elif remove_channels == "only_stoch":
            rm_mechanisms = stochastic_mechs
        elif remove_channels == "only_non_stoch":
            rm_mechanisms = non_stochastic_mechs
        else:
            logging.warning("Unknown GJ remove_channels setting: %s", remove_channels)
            rm_mechanisms = []
        if rm_mechanisms:
            logging.info("Remove channels type = %s", remove_channels)
            _perform_remove_channels(node_manager, rm_mechanisms)

    # load g_pas
    if filename := settings.get("load_g_pas_file"):
        processed_cells = _update_gpas(
            node_manager, filename, gjc, settings.get("correction_iteration_load", -1)
        )
        all_ranks_total = int(MPI.py_sum(processed_cells, 0))
        logging.info(
            "Update g_pas to fit %s - file %s for %s cells", gjc, filename, all_ranks_total
        )

    # load current clamps
    holding_ic_per_gid = {}
    if filename := settings.get("manual_MEComboInfo_file"):
        # Oren's note: If I manually injecting different holding current for each cell,
        # I will inject the current - the holding the emMEComboInfoFile
        if settings.get("procedure_type") == "find_holding_current":
            raise ConfigurationError("find_holding_current should not read manual_MEComboInfo_file")
        holding_ic_per_gid = _load_holding_ic(node_manager, filename, gjc=gjc)
        all_ranks_total = int(MPI.py_sum(len(holding_ic_per_gid), 0))
        logging.info(
            "Load holding_ic from manual_MEComboInfoFile %s for %s cells", filename, all_ranks_total
        )

    seclamp_per_gid = {}
    if settings.get("procedure_type") == "find_holding_current":
        seclamp_per_gid = _find_holding_current(node_manager, settings.get("vc_amp"))
        all_ranks_total = int(MPI.py_sum(len(seclamp_per_gid), 0))
        logging.info(
            "Inject holding voltages from file %s for %s cells",
            settings["vc_amp"],
            all_ranks_total,
        )

    return holding_ic_per_gid, seclamp_per_gid


def _update_conductance(gjc, gj_manager):
    """Update gap junction connection conductance"""
    n_conn = 0
    for conn in gj_manager.all_connections():
        conn.update_conductance(gjc)
        n_conn += 1
    return n_conn


def _deterministic_stoch(node_manager):
    """Enable deterministic_StochKV in cell"""
    for cell in node_manager.cells:
        for sec in cell._cellref.all:
            if "StochKv3" in dir(sec(0.5)):
                sec.deterministic_StochKv3 = 1
            if "StochKv2" in dir(sec(0.5)):
                sec.deterministic_StochKv2 = 1
            if "StochKv1" in dir(sec(0.5)):
                sec.deterministic_StochKv1 = 1


def _perform_remove_channels(node_manager, mechanisms: list):
    """Remove certain mechanisms from the cell"""
    for cell in node_manager.cells:
        for sec in cell._cellref.all:
            for mec in mechanisms:
                if mec in dir(sec(0.5)):
                    sec.uninsert(mec)


def _update_gpas(node_manager, filename, gjc, correction_iteration_load):
    """Update the g_pas attribute of certain cells
    Cell ids, segment names and new g_pas values are provided in the input file
    """
    import h5py

    processed_cells = 0
    try:
        g_pas_file = h5py.File(filename, "r")
    except OSError as e:
        raise ConfigurationError(f"Error opening g_pas file {filename}") from e
    raw_cell_gids = node_manager.local_nodes.raw_gids()
    offset = node_manager.local_nodes.offset
    if f"g_pas/{gjc}" not in g_pas_file:
        logging.warning("Data for g_pas/%s not found in %s", gjc, filename)
        return 0
    for agid in g_pas_file[f"g_pas/{gjc}/"]:
        gid = int(agid[1:])
        if gid in raw_cell_gids:  # if the node has a part of the cell
            cell = node_manager.getCell(gid + offset)
            processed_cells += 1
            for sec in cell.all:
                for seg in sec:
                    try:
                        attr_name = str(seg)[str(seg).index(".") + 1 :]
                        value = g_pas_file[f"g_pas/{gjc}/{agid}"][attr_name][
                            correction_iteration_load
                        ]
                    except Exception as e:
                        raise ConfigurationError(
                            f"Failed to load data in g_pas file {filename}: {e}"
                        ) from e
                    seg.g_pas = value
    return processed_cells


def _load_holding_ic(node_manager, filename, gjc):
    """Add IClamps to certain cells for holding the membrane voltage
    Cell ids, current amplitudes are provided in the input file
    """
    import h5py

    holding_ic_per_gid = {}
    try:
        holding_per_gid = h5py.File(filename, "r")
    except OSError as e:
        raise ConfigurationError(f"Error opening MEComboInfo file {filename}") from e

    if f"holding_per_gid/{gjc}" not in holding_per_gid:
        logging.warning("Data for holding_per_gid/%s not found in %s", gjc, holding_per_gid)
        return holding_ic_per_gid
    raw_cell_gids = node_manager.local_nodes.raw_gids()
    offset = node_manager.local_nodes.offset
    for agid in holding_per_gid["holding_per_gid"][str(gjc)]:
        gid = int(agid[1:])
        if gid in raw_cell_gids:
            final_gid = gid + offset
            holding_ic_per_gid[final_gid] = Nd.h.IClamp(
                0.5, sec=node_manager.getCell(final_gid).soma[0]
            )
            holding_ic_per_gid[final_gid].dur = 9e9
            try:
                holding_ic_per_gid[final_gid].amp = holding_per_gid["holding_per_gid"][str(gjc)][agid][()]  # noqa: E501 #fmt: skip
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load data in g_pas file {filename}: {e}"
                ) from e
    return holding_ic_per_gid


def _find_holding_current(node_manager, filename):
    """Add SEClamps to certain cells for holding the membrane voltage
    Cell ids, voltage amplitude are provided in the input file
    """
    import h5py

    try:
        v_per_gid = h5py.File(filename, "r")
    except OSError as e:
        raise ConfigurationError(f"Error opening voltage file {filename}") from e

    logging.info("Inject voltage clamps without disabling holding current!")

    seclamp_per_gid = {}
    raw_cell_gids = node_manager.local_nodes.raw_gids()
    offset = node_manager.local_nodes.offset
    for agid in v_per_gid["v_per_gid"]:
        gid = int(agid[1:])
        if gid in raw_cell_gids:
            final_gid = gid + offset
            seclamp_per_gid[final_gid] = Nd.h.SEClamp(
                0.5, sec=node_manager.getCell(final_gid).soma[0]
            )
            seclamp_per_gid[final_gid].dur1 = 9e9
            seclamp_per_gid[final_gid].amp1 = float(v_per_gid["v_per_gid"][agid][()])
            seclamp_per_gid[final_gid].rs = 0.0000001
    return seclamp_per_gid
