"""Tests of skipping the connections of virtual sources that no replay input can activate"""

import json
from types import SimpleNamespace

import libsonata
import numpy as np
import numpy.testing as npt
import pytest

from tests.replay_circuit import (
    BIO_TO_BIO,
    EXT_TO_BIO,
    N_BIO,
    VIRT_CONNECTIONS,
    kept_connections,
    make_replay_circuit,
)

from neurodamus import Neurodamus
from neurodamus.core import NeuronWrapper as Nd
from neurodamus.core.configuration import Feature, SimConfig
from neurodamus.io.sonata_config import ConnectionTypes
from neurodamus.node import Node
from neurodamus.replay import SpikeManager


@pytest.fixture
def circuit(tmp_path):
    return make_replay_circuit(tmp_path)


def virt_connections(sources, targets=range(N_BIO)):
    """The Virt -> Bio connections expected when only `sources` can be activated"""
    return {
        (src, tgt): n_syns
        for (src, tgt), n_syns in VIRT_CONNECTIONS.items()
        if src in sources and tgt in targets
    }


def build(circuit, spikes_files=(), **kwargs):
    config = circuit.write_simulation_config(spikes_files, **kwargs.pop("extra_config", {}))
    kwargs.setdefault("disable_reports", True)
    return Neurodamus(config, **kwargs)


# ----------------------------------------------------------------------
# SpikeManager.read_spiking_gids
# ----------------------------------------------------------------------


def test_read_spiking_gids_unique_and_sorted(circuit):
    spikes_file = circuit.spikes_file("spikes", "Virt", [4, 0, 3, 3], times=(1.0, 2.0, 3.0))
    gids = SpikeManager.read_spiking_gids(spikes_file, "Virt")
    npt.assert_array_equal(gids, [0, 3, 4])
    assert gids.dtype == np.int64


def test_read_spiking_gids_missing_population_is_empty(circuit):
    spikes_file = circuit.spikes_file("spikes", "Virt", [1, 2])
    gids = SpikeManager.read_spiking_gids(spikes_file, "Ext")
    assert gids.dtype == np.int64
    assert len(gids) == 0


def test_read_spiking_gids_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        SpikeManager.read_spiking_gids(str(tmp_path / "missing.h5"), "Virt")


# ----------------------------------------------------------------------
# Which connections are created
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "replayed",
    [
        [0, 3],  # connections with several synapses are kept whole
        [1, 2, 4, 5],  # sources that connect to several cells
        [3],
        [0, 1, 2, 3, 4, 5],  # all the sources: nothing is skipped
        [7, 9],  # sources without any connection
    ],
)
@pytest.mark.forked
def test_only_replayed_sources_are_connected(circuit, replayed):
    nd = build(circuit, [circuit.spikes_file("spikes", "Virt", replayed)])
    assert kept_connections(nd, "Virt") == virt_connections(replayed)


@pytest.mark.forked
def test_other_populations_are_not_affected(circuit):
    nd = build(circuit, [circuit.spikes_file("spikes", "Virt", [0])])
    # Ext is virtual and never replayed: no connection is created
    assert kept_connections(nd, "Ext") == {}
    # The connections between the real cells are always created
    assert set(kept_connections(nd, "Bio")) == set(BIO_TO_BIO)


@pytest.mark.forked
def test_all_populations_replayed(circuit):
    spikes_files = [
        circuit.spikes_file("virt", "Virt", [0, 1]),
        circuit.spikes_file("ext", "Ext", [1, 2]),
    ]
    nd = build(circuit, spikes_files)
    assert kept_connections(nd, "Virt") == virt_connections([0, 1])
    assert set(kept_connections(nd, "Ext")) == {
        (src, tgt) for src, tgt in EXT_TO_BIO if src in {1, 2}
    }


@pytest.mark.forked
def test_replay_inputs_are_added_up(circuit):
    spikes_files = [
        circuit.spikes_file("first", "Virt", [0]),
        circuit.spikes_file("second", "Virt", [4]),
        circuit.spikes_file("third", "Virt", [0, 5]),  # overlaps with the first
    ]
    nd = build(circuit, spikes_files)
    assert kept_connections(nd, "Virt") == virt_connections([0, 4, 5])


@pytest.mark.forked
def test_no_replay_input_skips_all_virtual_connections(circuit):
    nd = build(circuit)
    assert kept_connections(nd, "Virt") == {}
    assert kept_connections(nd, "Ext") == {}
    assert set(kept_connections(nd, "Bio")) == set(BIO_TO_BIO)


@pytest.mark.forked
def test_inputs_that_are_not_replay_are_ignored(circuit):
    pulse = {
        "module": "pulse",
        "input_type": "current_clamp",
        "delay": 1,
        "duration": 5,
        "node_set": "Bio",
        "represents_physical_electrode": True,
        "amp_start": 1,
        "width": 1,
        "frequency": 50,
    }
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    config = circuit.simulation_config([spikes_file])
    config["inputs"]["pulse"] = pulse
    config_file = circuit.directory / "with_pulse.json"
    config_file.write_text(json.dumps(config))
    nd = Neurodamus(str(config_file), disable_reports=True)
    assert kept_connections(nd, "Virt") == virt_connections([3])


@pytest.mark.forked
def test_replay_spikes_of_another_population_do_not_activate_sources(circuit):
    # The spikes are of a population that is not the source of the projection
    nd = build(circuit, [circuit.spikes_file("ext", "Ext", [0, 1, 2])])
    assert kept_connections(nd, "Virt") == {}
    assert len(kept_connections(nd, "Ext")) == 3


@pytest.mark.forked
def test_kept_synapses_keep_their_ids(circuit):
    """The synapse ids are the rows of the target cell, which seed the random number generators.

    Skipping a connection must not shift the ids of the others.
    """
    nd = build(circuit, [circuit.spikes_file("spikes", "Virt", [3])])
    manager = nd.circuits.get_edge_manager("Virt", "Bio")
    ids = {
        (conn.sgid - manager.src_pop_offset, conn.tgid - manager.target_pop_offset): [
            int(syn_id) for syn_id in conn._synapse_ids
        ]
        for conn in manager.all_connections()
    }
    # The skipped connection (0, 1) is the first row of the cell 1, followed by (3, 1)
    assert ids == {(3, 1): [1, 2, 3]}


# ----------------------------------------------------------------------
# Spontaneous minis
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_minis, inh_minis, cells_with_minis",
    [
        ([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], {0}),
        ([0.0, 0.0, 0.0], [0.0, 0.2, 0.0], {1}),
        ([0.1, 0.0, 0.0], [0.0, 0.2, 0.3], {0, 1, 2}),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], set()),  # zero rates are no minis
    ],
)
@pytest.mark.forked
def test_cells_with_minis_keep_all_their_connections(
    tmp_path, exc_minis, inh_minis, cells_with_minis
):
    circuit = make_replay_circuit(tmp_path, exc_minis=exc_minis, inh_minis=inh_minis)
    nd = build(circuit, [circuit.spikes_file("spikes", "Virt", [3])])
    expected = {
        key: n_syns
        for key, n_syns in VIRT_CONNECTIONS.items()
        if key[1] in cells_with_minis or key[0] == 3
    }
    assert kept_connections(nd, "Virt") == expected
    # The unreplayed Ext population follows the same rule
    expected_ext = {(src, tgt) for src, tgt in EXT_TO_BIO if tgt in cells_with_minis}
    assert set(kept_connections(nd, "Ext")) == expected_ext


# ----------------------------------------------------------------------
# Connection overrides
# ----------------------------------------------------------------------


def override(name="override", source="Virt", target="Bio", **kwargs):
    return {"connection_overrides": [{"name": name, "source": source, "target": target, **kwargs}]}


@pytest.mark.parametrize("rate", [0.01, 5.0])
@pytest.mark.forked
def test_spont_minis_override_keeps_all_connections(circuit, rate):
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    nd = build(
        circuit,
        [spikes_file],
        extra_config=override(spont_minis=rate),
        restrict_features=[Feature.Replay, Feature.SpontMinis],
    )
    assert kept_connections(nd, "Virt") == VIRT_CONNECTIONS


@pytest.mark.forked
def test_spont_minis_override_of_another_projection_disables_skipping_everywhere(circuit):
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    nd = build(
        circuit,
        [spikes_file],
        extra_config=override(source="Ext", spont_minis=0.5),
        restrict_features=[Feature.Replay, Feature.SpontMinis],
    )
    assert kept_connections(nd, "Virt") == VIRT_CONNECTIONS


@pytest.mark.forked
def test_zero_spont_minis_override_still_skips(circuit):
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    nd = build(
        circuit,
        [spikes_file],
        extra_config=override(spont_minis=0.0),
        restrict_features=[Feature.Replay, Feature.SpontMinis],
    )
    assert kept_connections(nd, "Virt") == virt_connections([3])


@pytest.mark.forked
def test_disabled_spont_minis_feature_ignores_the_override(circuit):
    """Without the SpontMinis feature neurodamus drops the rate, so there are no minis"""
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    nd = build(
        circuit,
        [spikes_file],
        extra_config=override(spont_minis=0.5),
        restrict_features=[Feature.Replay],
    )
    assert kept_connections(nd, "Virt") == virt_connections([3])


@pytest.fixture
def some_virt_node_set(circuit):
    """A node set with some of the Virt sources, for the connection overrides to select"""
    node_sets = json.loads(circuit.node_sets_file.read_text())
    node_sets["SomeVirt"] = {"population": "Virt", "node_id": [0, 1, 3]}
    circuit.node_sets_file.write_text(json.dumps(node_sets))


@pytest.mark.parametrize(
    "extra_config, replayed, connected_sources",
    [
        # The override takes all the sources and only skipping reduces the connections
        (override(weight=2.0), [0, 3, 4], {0, 3, 4}),
        # The override selects the sources 0, 1 and 3. Of those only 0 and 3 are replayed
        (override(source="SomeVirt", weight=1.5), [0, 3, 4], {0, 3}),
        # Everything selected by the override is silent
        (override(source="SomeVirt", weight=1.5), [4, 5], set()),
    ],
)
@pytest.mark.forked
def test_connections_created_by_overrides_are_skipped_too(
    circuit, some_virt_node_set, extra_config, replayed, connected_sources
):
    """With connection overrides, the connections are created by group instead of all at once"""
    spikes_file = circuit.spikes_file("spikes", "Virt", replayed)
    nd = build(circuit, [spikes_file], extra_config=extra_config)
    assert kept_connections(nd, "Virt") == virt_connections(connected_sources)


# ----------------------------------------------------------------------
# Other conditions that disable skipping
# ----------------------------------------------------------------------


@pytest.mark.forked
def test_synapse_report_keeps_all_connections(circuit):
    reports = {
        "synapse_report": {
            "type": "synapse",
            "cells": "Bio",
            "variable_name": "g",
            "sections": "all",
            "dt": 0.1,
            "start_time": 0.0,
            "end_time": 10.0,
        }
    }
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    nd = build(circuit, [spikes_file], extra_config={"reports": reports})
    assert kept_connections(nd, "Virt") == VIRT_CONNECTIONS


@pytest.mark.forked
def test_other_report_types_do_not_disable_skipping(circuit):
    reports = {
        "soma_v": {
            "type": "compartment",
            "cells": "Bio",
            "variable_name": "v",
            "sections": "soma",
            "dt": 0.1,
            "start_time": 0.0,
            "end_time": 10.0,
        }
    }
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    nd = build(circuit, [spikes_file], extra_config={"reports": reports})
    assert kept_connections(nd, "Virt") == virt_connections([3])


@pytest.mark.forked
def test_replay_feature_disabled_keeps_all_connections(circuit):
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    nd = build(circuit, [spikes_file], restrict_features=[Feature.Stimulus])
    assert kept_connections(nd, "Virt") == VIRT_CONNECTIONS


@pytest.fixture
def predicate_env(monkeypatch):
    """Make `Node._can_skip_unreplayed_connections` evaluate on a configurable fake environment"""
    env = SimpleNamespace(
        node=SimpleNamespace(_n_cycles=1),
        manager=SimpleNamespace(current_population=SimpleNamespace(virtual_source=True)),
        ptype=ConnectionTypes.Synaptic,
    )
    monkeypatch.setattr(
        SimConfig, "connections", [SimpleNamespace(spont_minis=None)], raising=False
    )
    monkeypatch.setattr(SimConfig, "reports", {}, raising=False)
    monkeypatch.setattr(
        SimConfig, "cli_options", SimpleNamespace(restrict_features=list(Feature)), raising=False
    )
    return env


def can_skip(env):
    return Node._can_skip_unreplayed_connections(env.node, env.ptype, env.manager)


def test_can_skip_by_default(predicate_env):
    assert can_skip(predicate_env)


@pytest.mark.parametrize(
    "ptype", [ConnectionTypes.GapJunction, ConnectionTypes.NeuroModulation, ConnectionTypes.Exp2Syn]
)
def test_cannot_skip_other_connection_types(predicate_env, ptype):
    predicate_env.ptype = ptype
    assert not can_skip(predicate_env)


def test_cannot_skip_connections_of_real_sources(predicate_env):
    predicate_env.manager.current_population.virtual_source = False
    assert not can_skip(predicate_env)


def test_cannot_skip_with_multiple_cycles(predicate_env):
    predicate_env.node._n_cycles = 2
    assert not can_skip(predicate_env)


def test_cannot_skip_without_the_replay_feature(predicate_env):
    SimConfig.cli_options.restrict_features = [Feature.Stimulus]
    assert not can_skip(predicate_env)


@pytest.mark.parametrize(
    "rates, expected", [([None, 0.0], True), ([None, 0.1], False), ([1.0], False)]
)
def test_spont_minis_overrides_disable_skipping(predicate_env, rates, expected):
    SimConfig.connections = [SimpleNamespace(spont_minis=rate) for rate in rates]
    assert can_skip(predicate_env) is expected


def test_no_connection_overrides(predicate_env):
    SimConfig.connections = []
    assert can_skip(predicate_env)


@pytest.mark.parametrize("report_type", ["compartment", "summation", "lfp"])
def test_reports_that_do_not_disable_skipping(predicate_env, report_type):
    kind = getattr(libsonata.SimulationConfig.Report.Type, report_type)
    SimConfig.reports = {"report": SimpleNamespace(type=kind)}
    assert can_skip(predicate_env)


def test_synapse_report_disables_skipping_among_others(predicate_env):
    types = libsonata.SimulationConfig.Report.Type
    SimConfig.reports = {
        "soma": SimpleNamespace(type=types.compartment),
        "syn": SimpleNamespace(type=types.synapse),
    }
    assert not can_skip(predicate_env)


# ----------------------------------------------------------------------
# The helper that drops the ranges
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_minis, inh_minis, replayed_srcs, kept_starts",
    [
        (None, None, [0, 3], [0, 5]),
        (None, None, [7], []),  # none of the connections is replayed
        (None, None, [], []),  # nothing is replayed
        (None, None, None, [0, 2, 3, 5]),  # not restricted
        (0.1, None, [], [0, 2, 3, 5]),  # cell with exc minis
        (None, 0.1, [], [0, 2, 3, 5]),  # cell with inh minis
        (0.0, 0.0, [1], [2]),  # zero rates are no minis
    ],
)
def test_drop_silent_ranges(exc_minis, inh_minis, replayed_srcs, kept_starts):
    from neurodamus.connection_manager import ConnectionManagerBase

    cell = SimpleNamespace(exc_mini_frequency=exc_minis, inh_mini_frequency=inh_minis)
    manager = SimpleNamespace(
        _cell_manager=SimpleNamespace(get_cell=lambda _: cell),
        _cur_population=SimpleNamespace(replayed_srcs=replayed_srcs),
    )
    # Four connections from sgids 0, 1, 2, 3, with 2, 1, 2, 1 synapses
    sgids = np.array([0, 0, 1, 2, 2, 3])
    ranges = [(0, 2), (2, 3), (3, 5), (5, 6)]

    kept = ConnectionManagerBase._drop_silent_ranges(manager, ranges, sgids, 7)

    assert [start for start, _ in kept] == kept_starts


def test_drop_silent_ranges_without_ranges():
    from neurodamus.connection_manager import ConnectionManagerBase

    cell = SimpleNamespace(exc_mini_frequency=None, inh_mini_frequency=None)
    manager = SimpleNamespace(
        _cell_manager=SimpleNamespace(get_cell=lambda _: cell),
        _cur_population=SimpleNamespace(replayed_srcs=np.array([1, 2])),
    )
    assert ConnectionManagerBase._drop_silent_ranges(manager, [], np.array([]), 7) == []


def test_unrestricted_population_does_not_look_at_the_cell():
    """Connections that are not restricted are never filtered, nor do they need the cells"""
    from neurodamus.connection_manager import ConnectionManagerBase

    def fail(_):
        raise AssertionError("The cell must not be looked up")

    manager = SimpleNamespace(
        _cell_manager=SimpleNamespace(get_cell=fail),
        _cur_population=SimpleNamespace(replayed_srcs=None),
    )
    ranges = [(0, 1)]
    assert ConnectionManagerBase._drop_silent_ranges(manager, ranges, np.array([0]), 0) is ranges


# ----------------------------------------------------------------------
# The simulation is not changed
# ----------------------------------------------------------------------


def run_and_record(nd):
    """Run the simulation recording the voltage of the soma of every cell"""
    manager = nd.circuits.get_node_manager("Bio")
    voltages = {}
    for gid in manager.local_nodes.gids(raw_gids=False):
        voltage = Nd.Vector()
        voltage.record(manager.get_cell(int(gid))._cellref.soma[0](0.5)._ref_v)
        voltages[int(gid)] = voltage
    Nd.finitialize()
    nd.run(cleanup=False)
    return {gid: voltage.as_numpy().copy() for gid, voltage in voltages.items()}


def build_and_run(config, monkeypatch, skip):
    skipper = Node._can_skip_unreplayed_connections
    monkeypatch.setattr(
        Node, "_can_skip_unreplayed_connections", lambda *args: skip and skipper(*args)
    )
    nd = Neurodamus(config, disable_reports=True)
    voltages = run_and_record(nd)
    connections = {pop: kept_connections(nd, pop) for pop in ("Virt", "Ext")}
    nd.clear_model()
    Nd.t = 0.0  # The replay ignores the spikes before the current time
    return voltages, connections


@pytest.mark.parametrize("replayed", [[0, 3], [1, 2, 4], [5]])
@pytest.mark.forked
def test_skipping_does_not_change_the_simulation(circuit, monkeypatch, replayed):
    spikes_file = circuit.spikes_file("spikes", "Virt", replayed, times=(2.0, 5.0, 9.0, 15.0))
    config = circuit.write_simulation_config([spikes_file])

    skipped, skipped_conns = build_and_run(config, monkeypatch, skip=True)
    full, full_conns = build_and_run(config, monkeypatch, skip=False)

    assert skipped_conns["Virt"] == virt_connections(replayed)
    assert full_conns["Virt"] == VIRT_CONNECTIONS
    assert skipped_conns["Ext"] == {}
    assert len(full_conns["Ext"]) == 3
    assert skipped.keys() == full.keys()
    for gid, voltage in full.items():
        npt.assert_array_equal(skipped[gid], voltage)
    # The replay does have an effect
    assert max(np.ptp(voltage) for voltage in full.values()) > 0.5


@pytest.mark.forked
def test_skipping_does_not_change_the_simulation_with_minis(tmp_path, monkeypatch):
    circuit = make_replay_circuit(tmp_path, exc_minis=[0.0, 5.0, 0.0], inh_minis=[0.0, 0.0, 0.0])
    spikes_file = circuit.spikes_file("spikes", "Virt", [0, 4], times=(2.0, 9.0))
    config = circuit.write_simulation_config([spikes_file])

    skipped, skipped_conns = build_and_run(config, monkeypatch, skip=True)
    full, full_conns = build_and_run(config, monkeypatch, skip=False)

    # The cell with minis (1) keeps all its connections. The others lose the silent ones
    assert {tgt for _, tgt in skipped_conns["Virt"]} == {0, 1, 2}
    assert {key for key in skipped_conns["Virt"] if key[1] == 1} == {
        key for key in VIRT_CONNECTIONS if key[1] == 1
    }
    assert len(skipped_conns["Virt"]) < len(full_conns["Virt"])
    for gid, voltage in full.items():
        npt.assert_array_equal(skipped[gid], voltage)


# ----------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------


@pytest.mark.forked
def test_dry_run_still_counts_every_synapse(circuit):
    spikes_file = circuit.spikes_file("spikes", "Virt", [3])
    config = circuit.write_simulation_config([spikes_file])
    nd = Neurodamus(config, dry_run=True, num_target_ranks=1)
    nd.run()
    n_synapses = nd._dry_run_stats.synapse_counts[ConnectionTypes.Synaptic]
    n_total = sum(VIRT_CONNECTIONS.values()) + len(EXT_TO_BIO) + len(BIO_TO_BIO)
    assert n_synapses == pytest.approx(n_total, rel=0.05)
