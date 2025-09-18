import json

import numpy as np

import pytest

from neurodamus.core.nodeset import SelectionNodeSet
from neurodamus.target_manager import NodeSetReader

import libsonata


@pytest.mark.forked
def test_SelectionNodeSet_base():
    # No registration, just plain gid sets
    set1 = SelectionNodeSet([1, 2, 3])
    assert set1.offset == 0
    assert set1.max_gid == 3

    set2 = SelectionNodeSet()
    assert set2.offset == 0
    assert set2.max_gid == 0
    set2.add_gids([1, 2, 3])
    assert set2.offset == 0
    assert set2.max_gid == 3


@pytest.mark.forked
def test_SelectionNodeSet_add():
    set_mid = SelectionNodeSet([1, 2, 3, 1000]).register_global("pop2")
    assert set_mid.offset == 0
    assert set_mid.max_gid == 1000

    # Append to right
    set_right = SelectionNodeSet([1, 2, 3, 4]).register_global("pop3")
    assert set_right.offset == 1000
    assert set_right.max_gid == 4

    # Append to left (occupies two blocks of 1000)
    set_left = SelectionNodeSet([1, 2, 3, 4, 1001]).register_global("pop1")
    assert set_left.offset == 0
    assert set_left.max_gid == 1001
    assert set_mid.offset == 2000
    assert set_right.offset == 3000

    # Extend middle to also occupy two blocks (rare)
    set_mid.add_gids([1002])
    assert set_mid.max_gid == 1002
    assert set_right.offset == 4000

@pytest.fixture
def nodeset_files(tmpdir):
    """
    Generates example nodeset files
    """
    # Define nodesets
    config_nodeset_file = {
        "Mosaic": {
            "population": "default"
        },
        "Layer1": {
            "layer": 1
        },
    }

    simulation_nodesets_file = {
        "Mosaic": {
            "population": "All"
        },
        "Layer2": {
            "layer": 2
        },
    }

    # Create the test JSON files with nodesets
    config_nodeset = tmpdir.join("node_sets_circuit.json")
    with open(config_nodeset, 'w') as f:
        json.dump(config_nodeset_file, f)

    simulation_nodeset = tmpdir.join("node_sets_simulation.json")
    with open(simulation_nodeset, 'w') as f:
        json.dump(simulation_nodesets_file, f)

    return config_nodeset, simulation_nodeset

@pytest.mark.forked
def test_read_nodesets_from_file(nodeset_files):
    ns_reader = NodeSetReader(*nodeset_files)
    assert ns_reader.names == {'Mosaic', 'Layer1', 'Layer2'}
    assert ns_reader.read_nodeset("Mosaic") is not None
    assert ns_reader.read_nodeset("Layer3") is None
    expected_output = {
        "Layer1": {"layer": [1]},
        "Layer2": {"layer": [2]},
        "Mosaic": {"population": ["All"]}
    }
    assert json.loads(ns_reader.nodesets.toJSON()) == json.loads(json.dumps(expected_output))

@pytest.mark.forked
def test_from_zero_based_libsonata_selection_invalid_argument():
    with pytest.raises(TypeError, match="Expected libsonata.Selection"): 
        SelectionNodeSet.from_zero_based_libsonata_selection("wrong")

@pytest.mark.forked
def test_selection():
    sel = libsonata.Selection([(3, 9), (11, 12)])
    ref = SelectionNodeSet(sel)
    offset = 3
    ref._offset = offset
    assert sel == ref.selection(raw_gids=True)
    sel = libsonata.Selection([(start+offset, stop+offset) for start, stop in sel.ranges])
    assert sel == ref.selection(raw_gids=False)

@pytest.mark.forked
def test_intersection_basic():
    a = SelectionNodeSet([1, 2, 3])
    a.register_global("pop")
    b = SelectionNodeSet([2, 3, 4])
    b.register_global("pop") 
    result = a.intersection(b).flatten()
    assert np.array_equal(result, np.array([2, 3], dtype=np.uint32))

@pytest.mark.forked
def test_intersection_different_population():
    a = SelectionNodeSet([1, 2, 3])
    a.register_global("popA")
    b = SelectionNodeSet([2, 3, 4])
    b.register_global("popB") 
    result = a.intersection(b)
    assert not result

@pytest.mark.forked
def test_intersection_wrong_type():
    a = SelectionNodeSet([1, 2, 3])
    with pytest.raises(TypeError):
        a.intersection([2, 3])  # not a SelectionNodeSet

@pytest.mark.forked
def test_intersection_with_offset():
    a = SelectionNodeSet([1, 2, 3])
    a.register_global("pop")
    a._offset = 10
    b = SelectionNodeSet([2, 3, 4])
    b.register_global("pop") 
    b._offset = 10
    result = a.intersection(b).flatten()
    assert np.array_equal(result, np.array([12, 13], dtype=np.uint32))


    
