import numpy.testing as npt

from neurodamus.core.nodeset import SelectionNodeSet
from neurodamus.target_manager import TargetSpec

null_target_spec = TargetSpec(None, None)


def test_targetspec_overlap_name():
    assert null_target_spec.overlap_byname(null_target_spec)
    t1_spec = TargetSpec("t1", None)
    assert t1_spec.overlap_byname(null_target_spec)
    assert null_target_spec.overlap_byname(t1_spec)
    assert not t1_spec.overlap_byname(TargetSpec("t2", None))
    assert not t1_spec.overlap_byname(TargetSpec("t2", "another"))
    # With populations, should ignore
    assert TargetSpec("t1", "popx").overlap_byname(TargetSpec("t1", "popy"))
    assert not TargetSpec("t1", "popx").overlap_byname(TargetSpec("t2", "popy"))


def test_populations_disjoint():
    """We test for disjoint since population doesnt say much about intersection,
    it can simply help identifying when they are disjoint for sure
    """
    assert not null_target_spec.disjoint_populations(null_target_spec)
    assert TargetSpec(None, "pop1").disjoint_populations(TargetSpec(None, "pop2"))
    assert not TargetSpec(None, "pop1").disjoint_populations(TargetSpec(None, "pop1"))


def test_targetspec_overlap():
    assert null_target_spec.overlap(null_target_spec)
    assert not TargetSpec("x", "pop1").overlap(null_target_spec)  # we cant be sure -> false
    assert not TargetSpec("target1", "pop1").overlap(TargetSpec("target2", None))
    assert not TargetSpec(None, "pop1").overlap(TargetSpec(None, "pop2"))  # disjoint pop
    assert TargetSpec(None, "pop1").overlap(TargetSpec("ahh", "pop1"))
    assert not TargetSpec("t1", "pop1").overlap(TargetSpec("t2", "pop1"))


def test_nodeset_target_intersect():
    from neurodamus.target_manager import NodesetTarget
    nodes_popA = SelectionNodeSet([1, 2]).register_global("pop_A")
    nodes2_popA = SelectionNodeSet([2, 3]).register_global("pop_A")
    nodes3_popA = SelectionNodeSet([11, 12]).register_global("pop_A")
    nodes_popB = SelectionNodeSet([1, 2]).register_global("pop_B")

    t1 = NodesetTarget("t1", [nodes_popA])
    t2 = NodesetTarget("t2", [nodes_popB])
    assert not t1.intersects(t2)
    t2.nodesets = [nodes_popB, nodes_popA]
    assert t1.intersects(t2)
    t2.nodesets = [nodes_popB, nodes2_popA]
    assert t1.intersects(t2)
    t2.nodesets = [nodes3_popA]
    assert not t1.intersects(t2)
    t2.nodesets = [nodes_popB, nodes3_popA]
    assert not t1.intersects(t2)


def test_nodeset_gids():
    from neurodamus.target_manager import NodesetTarget
    local_nodes_popA = SelectionNodeSet(range(5, 10)).register_global("pop_A")
    local_nodes_popB = SelectionNodeSet(range(6)).register_global("pop_B")
    nodes_popA = SelectionNodeSet(range(7)).register_global("pop_A")
    nodes_popB = SelectionNodeSet([3, 4, 5]).register_global("pop_B")
    t = NodesetTarget(
        "target-a",
        [nodes_popA, nodes_popB],
        local_nodes=[local_nodes_popA, local_nodes_popB]
    )
    gids = t.get_local_gids()
    npt.assert_array_equal(gids, [5, 6, 1003, 1004, 1005])

    t2 = NodesetTarget(
        "target-b",
        [nodes_popA, nodes_popB],
        local_nodes=[local_nodes_popA]
    )
    gids = t2.get_local_gids()
    npt.assert_array_equal(gids, [5, 6])

    t3 = NodesetTarget(
        "target-c",
        [nodes_popB],
        local_nodes=[local_nodes_popA, local_nodes_popB]
    )
    gids = t3.get_local_gids()
    npt.assert_array_equal(gids, [1003, 1004, 1005])

    t4 = NodesetTarget(
        "target-d",
        [nodes_popB],
        local_nodes=[local_nodes_popA]
    )
    gids = t4.get_local_gids()
    npt.assert_array_equal(gids, [])
