import numpy as np
import pytest

from neurodamus.utils.multimap import GroupedMultiMap


def test_basic():
    d = GroupedMultiMap(np.array([3, 1, 2, 1, 2], "i"), ["a", "b", "c", "d", "e"])
    with pytest.raises(KeyError):
        d[0]
    assert d.get(0, None) is None
    assert d[1] == ["b", "d"]
    assert len(d) == 3
    assert list(iter(d)) == [1, 2, 3]

    assert list(d.items()) == [(1, ["b", "d"]), (2, ["c", "e"]), (3, ["a"])]

    with pytest.raises(NotImplementedError):
        d[1] = "asdf"

    assert 1 in d


def test_map_duplicates():
    keys = np.array([3, 1, 2, 1, 2], "i")
    vals = ["a", "b", "c", "d", "e"]
    d = GroupedMultiMap(keys, vals)
    assert np.array_equal(d.keys(), [1, 2, 3])
    assert [list(v) for v in d.values()] == [["b", "d"], ["c", "e"], ["a"]]
    assert d[1] == ["b", "d"]
    assert d[2] == ["c", "e"]
    assert d[3] == ["a"]
    assert list(d.get_items(1)) == ["b", "d"]


def test_badly_formed():
    keys = np.array([3, 4], "i")
    vals = []
    with pytest.raises(AssertionError):
        GroupedMultiMap(keys, vals)

    d = GroupedMultiMap(np.array([], "i"), [])
    assert len(d.values()) == 0


def test_merge_grouped():
    d = GroupedMultiMap(np.array([3, 4, 3], "i"), [1, 2, 3])
    d += GroupedMultiMap(np.array([2, 4, 3], "i"), ["x", "y", "z"])
    assert d[3] == [1, 3, "z"]

    d = GroupedMultiMap(np.array([3, 4, 3], "i"), [1, 2, 3])
    d += GroupedMultiMap(np.array([2, 4, 3], "i"), ["x", "y", "z"])
    assert d[3] == [1, 3, "z"]
