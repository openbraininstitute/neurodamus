"""Compatibility Classes to work similar to HOC types, recreating or wrapping them"""

from array import array


class Vector(array):
    """Behavior similar to Hoc Vector"""

    __slots__ = ()

    def __new__(cls, type_="I", array=None):
        if array is None:
            array = []
        return super().__new__(Vector, type_, array)

    def size(self):
        return len(self)

    @property
    def x(self):
        return self

    def __add__(self, other):
        array.extend(self, other)
        return self


class List(list):  # noqa: FURB189
    """Behavior similar to Hoc List"""

    __slots__ = ()

    def count(self, obj=None):
        return super().count(obj) if obj else len(self)

    def o(self, idx):
        return self[int(idx)]


def hoc_vector(np_array):
    from neuron import h

    hoc_vec = h.Vector(np_array.size)
    hoc_vec.as_numpy()[:] = np_array
    return hoc_vec
