import numpy
import numpy as np
import pytest


@pytest.mark.forked
def test_get_local_gids():

    from neurodamus.core.nodeset import SelectionNodeSet, PopulationNodes
    from neurodamus.target_manager import NodesetTarget
    PopulationNodes.reset()
    nodes_popA = SelectionNodeSet([1, 2]).register_global("pop_A")
    nodes_popB = SelectionNodeSet([1, 2]).register_global("pop_B")
    local_gids = [SelectionNodeSet([1]).register_global("pop_A"), SelectionNodeSet([2]).register_global("pop_B")]
    t1 = NodesetTarget("t1", [nodes_popA], local_gids)
    t2 = NodesetTarget("t2", [nodes_popB], local_gids)
    t_empty = NodesetTarget("t_empty", [], local_gids)
    numpy.testing.assert_array_equal(t1.get_local_gids(), [1])
    numpy.testing.assert_array_equal(t1.get_local_gids(raw_gids=True), [1])
    numpy.testing.assert_array_equal(t2.get_local_gids(), [1002])
    numpy.testing.assert_array_equal(t2.get_local_gids(raw_gids=True), [2])
    numpy.testing.assert_array_equal(t_empty.get_local_gids(), [])


def test_nodeset_target_generate_subtargets():
    """Test NodesetTarget correctly partitions nodes into subtargets
    with expected GID distributions."""

    from neurodamus.core.nodeset import SelectionNodeSet
    from neurodamus.target_manager import NodesetTarget

    N_PARTS = 3
    raw_gids_a = list(range(10))
    raw_gids_b = list(range(5))
    nodes_popA = SelectionNodeSet(raw_gids_a).register_global("pop_A")
    nodes_popB = SelectionNodeSet(raw_gids_b).register_global("pop_B")
    target = NodesetTarget("Column", [nodes_popA, nodes_popB])
    assert np.array_equal(target.gids(raw_gids=False),
                          np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003, 1004]))

    subtargets = target.generate_subtargets(N_PARTS)
    assert len(subtargets) == N_PARTS
    assert len(subtargets[0]) == 2
    subtarget_popA = subtargets[0][0]
    subtarget_popB = subtargets[0][1]
    assert subtarget_popA.name == "pop_A__Column_0"
    assert subtarget_popA.population_names == {"pop_A"}
    assert np.array_equal(subtarget_popA.gids(raw_gids=False), np.array([0, 3, 6, 9]))
    assert subtarget_popB.name == "pop_B__Column_0"
    assert subtarget_popB.population_names == {"pop_B"}
    assert np.array_equal(subtarget_popB.gids(raw_gids=False), np.array([1000, 1003]))
    assert np.array_equal(subtargets[1][0].gids(raw_gids=False), np.array([1, 4, 7]))
    assert np.array_equal(subtargets[2][0].gids(raw_gids=False), np.array([2, 5, 8]))
    assert np.array_equal(subtargets[1][1].gids(raw_gids=False), np.array([1001, 1004]))
    assert np.array_equal(subtargets[2][1].gids(raw_gids=False), np.array([1002]))
