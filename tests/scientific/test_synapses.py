import re
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

USECASE3 = Path(__file__).parent.parent.absolute() / "simulations" / "usecase3"
SSCX_V7 = Path(__file__).parent.parent.absolute() / "simulations" / "sscx-v7-plasticity"


@pytest.mark.slow
def test_synapses_params():
    """
    A test of the impact of eager caching of synaptic parameters. BBPBGLIB-813
    """
    from neurodamus.core import NeuronWrapper as Nd
    from neurodamus.node import Node
    from neurodamus.core.configuration import GlobalConfig, SimConfig, LogLevel
    from neurodamus.io.synapse_reader import SynapseParameters
    from neurodamus.utils.logging import log_verbose
    from libsonata import EdgeStorage

    # create Node from config
    GlobalConfig.verbosity = LogLevel.VERBOSE
    config_file = str(SSCX_V7 / "simulation_config_base.json")
    n = Node(config_file)
    conn_weight = 0.8  # for testing

    # append Connection blocks programmatically
    # plasticity
    CONN_plast = {
        "Source": "pre_L5_PC",
        "Destination": "post_L5_PC",
        "ModOverride": "GluSynapse",
        "Weight": conn_weight
    }
    SimConfig.connections["plasticity"] = CONN_plast
    # init_I_E
    CONN_i2e = {
        "Source": "pre_L5_BC",
        "Destination": "post_L5_PC",
        "Weight": conn_weight
    }
    SimConfig.connections["init_I_E"] = CONN_i2e

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    # init
    base_seed = n._run_conf.get("BaseSeed", 0)  # base seed for synapse RNG
    for syn_manager in n._circuits.all_synapse_managers():
        syn_manager.finalize(base_seed)
    n.sim_init()

    # 1) get synapse parameters from libsonata
    target_manager = n.target_manager
    pop1, pre_L5_BC = get_target_raw_gids(target_manager, "pre_L5_BC")[0]
    pop2, post_L5_PC = get_target_raw_gids(target_manager, "post_L5_PC")[0]
    pop3, pre_L5_PC = get_target_raw_gids(target_manager, "pre_L5_PC")[0]
    assert pop1 == pop2
    dfs = {}
    properties = ["conductance", "u_syn", "decay_time", "depression_time", "facilitation_time",
                  "n_rrp_vesicles", "conductance_scale_factor", "u_hill_coefficient", "weight"]
    plast_params = ["volume_CR", "rho0_GB", "Use_d_TM", "Use_p_TM",
                    "gmax_d_AMPA", "gmax_p_AMPA", "theta_d", "theta_p", "gmax_NMDA"]

    edges_file, edge_pop = n._sonata_circuits[pop1].nrnPath.split(":")
    storage = EdgeStorage(edges_file)
    edge_pop = storage.open_population(edge_pop)
    sel1 = edge_pop.connecting_edges(pre_L5_BC, post_L5_PC)
    sel2 = edge_pop.connecting_edges(pre_L5_PC, post_L5_PC)
    df = get_edge_properties(edge_pop, sel1, properties)
    df["weight"] = df["conductance"] * conn_weight  # compute weight column
    dfs['ProbGABAAB_EMS'] = df
    df = get_edge_properties(edge_pop, sel2, properties+plast_params)
    df["gmax_NMDA"] = df["conductance"] * df["conductance_scale_factor"]  # compute gmax_NMDA column
    df["weight"] = 1.0  # compute weight column (not set in Connection block for GluSynapse)
    dfs['GluSynapse'] = df

    # scale Use with calcium
    # wrapper class for calling SonataReader._scale_U_param
    class wrapU:
        def __init__(self, a, b):
            self.U = np.array(a)
            self.u_hill_coefficient = np.array(b)
            assert (self.U.size == self.u_hill_coefficient.size)

        def __len__(self):
            return self.U.size

    for df in dfs.values():
        tmp = wrapU(df["u_syn"], df["u_hill_coefficient"])
        SynapseParameters._patch_scale_U_param(tmp, SimConfig.extracellular_calcium, [])
        df["u_syn"] = tmp.U

    # 2) get values from NEURON
    post_cell = n.circuits.global_manager.get_cellref(post_L5_PC + 1)  # 1-based in neurodamus
    # here we collect all synapses for the post cell
    match_index = re.compile(r"\[[0-9]+\]$")
    synlist = {}
    for nc in Nd.h.cvode.netconlist('', post_cell, ''):
        if nc.precell() is not None:  # minis netcons only
            continue
        syn = nc.syn()
        syntype = match_index.sub('', syn.hname())
        d = {'weight': nc.weight[0]}
        for v in vars(syn):
            try:
                attr = getattr(syn, v)
                if attr.__class__.__name__ in ['int', 'float', 'str']:
                    d[v] = attr
            except Exception:
                continue
        synlist.setdefault(syntype, []).append(d)

    # sort lists by synapseID
    for _, x in synlist.items():
        x.sort(key=lambda d: d['synapseID'])

    # 3) compare values: Neurodamus vs libsonata
    # mapping between Nd and libsonata properties
    properties = {
        'ProbAMPANMDA_EMS':
        {
            'conductance': 'conductance',
            'Dep': 'depression_time',
            'Fac': 'facilitation_time',
            'NMDA_ratio': 'conductance_scale_factor',
            'Nrrp': 'n_rrp_vesicles',
            'tau_d_AMPA': 'decay_time',
            'Use': 'u_syn',
            'weight': "weight"
        },
        'ProbGABAAB_EMS':
        {
            'conductance': 'conductance',
            'Dep': 'depression_time',
            'Fac': 'facilitation_time',
            'GABAB_ratio': 'conductance_scale_factor',
            'Nrrp': 'n_rrp_vesicles',
            'tau_d_GABAA': 'decay_time',
            'Use': 'u_syn',
            'weight': "weight"
        },
        'GluSynapse':
        {
            'Dep': 'depression_time',
            'Fac': 'facilitation_time',
            'gmax0_AMPA': 'conductance',
            'gmax_d_AMPA': "gmax_d_AMPA",
            'gmax_NMDA': "gmax_NMDA",
            'gmax_p_AMPA': "gmax_p_AMPA",
            'Nrrp': 'n_rrp_vesicles',
            'rho0_GB': "rho0_GB",
            'tau_d_AMPA': 'decay_time',
            'theta_d_GB': "theta_d",
            'theta_p_GB': "theta_p",
            'Use': 'u_syn',
            'volume_CR': "volume_CR",
            'weight': "weight"
        }
    }

    for stype, syns in synlist.items():
        for i, info in enumerate(syns):
            log_verbose("%s[%d] (ID %d)" % (stype, i, info['synapseID']))
            for prop, dfcol in properties[stype].items():
                log_verbose("    %12s %12.6f ~= %-12.6f %s" %
                            (prop, info[prop], dfs[stype][dfcol][i], dfcol))
                assert (info[prop] == pytest.approx(dfs[stype][dfcol][i]))


def get_edge_properties(edge_pop, selection, properties=[]):
    """Get an array of edge IDs or DataFrame with edge properties."""
    edge_ids = selection.flatten()
    if not properties:
        return edge_ids
    dtype = np.dtype({"names": properties, "formats": ["f8"] * len(properties)})
    if len(edge_ids) == 0:
        result = np.recarray(0, dtype)
    else:
        result = np.recarray(len(edge_ids), dtype)
        for p in properties:
            if p in edge_pop.attribute_names:
                result[p] = edge_pop.get_attribute(p, selection)
    return result


def test__constrained_hill():
    from neurodamus.io.synapse_reader import SynapseParameters

    # original functions
    def hill(ca_conc, y, K_half):
        return y*ca_conc**4/(K_half**4 + ca_conc**4)

    def constrained_hill(K_half):
        y_max = (K_half**4 + 16) / 16
        return lambda x: hill(x, y_max, K_half)

    f_scale = lambda x, y: constrained_hill(x)(y)
    scale_factors = np.vectorize(f_scale)

    rng = np.random.default_rng(seed=42)
    a = 10 * rng.random(100)
    b = 10 * rng.random(100)

    npt.assert_allclose(scale_factors(a, 2), SynapseParameters._constrained_hill(a, 2))
    npt.assert_allclose(scale_factors(a, 2.2), SynapseParameters._constrained_hill(a, 2.2))
    npt.assert_allclose(scale_factors(a, b), SynapseParameters._constrained_hill(a, b))


def get_target_raw_gids(target_manager, target_name):
    tgt = target_manager.get_target(target_name)
    return tuple(zip(tgt.population_names, tgt.get_raw_gids() - 1))  # 0-based


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_data": {
            "network": "no_edge_circuit_config.json",
            "node_sets_file": "nodesets.json",
            "run":
            {
                "random_seed": 12345,
                "dt": 0.05,
                "tstop": 1000
            }
        },
        "src_dir": USECASE3
    }
], indirect=True)
def test_no_edge_creation(capsys, create_tmp_simulation_config_file):
    from neurodamus.node import Node

    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.create_synapses()

    captured = capsys.readouterr()
    assert "No connectivity set as internal" in captured.out
    assert len(n.circuits.edge_managers) == 0
