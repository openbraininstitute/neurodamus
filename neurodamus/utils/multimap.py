"""A collection of Pure-Python MultiMaps"""

import collections.abc
from functools import reduce
from operator import add

import numpy as np


class GroupedMultiMap(collections.abc.Mapping):
    """A Multimap which groups values by key in a list."""

    __slots__ = (
        "_keys",  # stored as `np.array`
        "_values",  # stored as `list`
    )

    def __init__(self, np_keys, values):
        """Init GroupedMultiMap.

        Args:
            np_keys: The numpy array of the keys. Can be empty
            values: The array of the values, can be any indexable, but better if numpy
        """
        assert len(np_keys) == len(values), "Keys and values must have the same length"
        self._keys, self._values = self.sort_together(np_keys, values)
        self._keys, self._values = self._duplicates_to_list(self._keys, self._values)

    @staticmethod
    def sort_together(np_keys, values):
        sort_idxs = np_keys.argsort(kind="mergesort")  # need stability
        keys = np_keys[sort_idxs]
        if isinstance(values, np.ndarray):
            values = values[sort_idxs]
        else:
            values = [values[i] for i in sort_idxs]
        return keys, values

    def find(self, key):
        idx = np.searchsorted(self._keys, key)
        if idx == len(self._keys) or self._keys[idx] != key:
            return None
        return idx

    def keys(self):
        return self._keys

    def values(self):
        return self._values

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, key):
        idx = self.find(key)
        if idx is None:
            raise KeyError(f"{key} does not exist")
        return self._values[idx]

    def __setitem__(self, key, value):
        raise NotImplementedError(
            "Setitem is not allowed for performance reasons. "
            "Please build and add-inplace another MultiMap"
        )

    def items(self):
        return zip(self._keys, self._values)

    def __contains__(self, key):
        return self.find(key) is not None

    exists = __contains__  # Compat. w Hoc map

    @staticmethod
    def _concat(v1, v2):
        return (v1 if isinstance(v1, (list, tuple)) else list(v1)) + (
            v2 if isinstance(v2, (list, tuple)) else list(v2)
        )

    @staticmethod
    def _duplicates_to_list(np_keys, values):
        np_keys, indexes = np.unique(np_keys, return_index=True)
        if len(indexes) == 0:
            return np_keys, []
        beg_it = iter(indexes)
        end_it = iter(indexes)
        next(end_it)  # Discard first
        values = [values[next(beg_it) : end] for end in end_it] + [values[indexes[-1] :]]  # Last
        assert len(np_keys) == len(values)
        return np_keys, values

    def get(self, key, default=()):
        idx = self.find(key)
        if idx is None:
            return default
        return self._values[idx]

    def get_items(self, key):
        return self.get(key)

    def __iadd__(self, other):
        """Inplace add (incorporate other)"""
        self._keys, self._values = self.sort_together(
            np.concatenate((self._keys, other._keys)), self._concat(self._values, other._values)
        )
        self._keys, v_list = self._duplicates_to_list(self._keys, self._values)
        self._values = [reduce(add, subl) for subl in v_list]
        return self
