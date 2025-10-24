"""Implementation of Gid Sets with the ability of self offsetting and avoid
global overlapping
"""

from __future__ import annotations

from contextlib import contextmanager

import libsonata
import numpy as np

from . import MPI
from neurodamus.utils.pyutils import WeakList


class PopulationNodes:
    """Handle SelectionNodeSets belonging to a population. Given that Neuron doesnt
    inherently handle populations, we will have to apply gid offsetting.
    The class stores `SelectionNodeSet`s, and makes the required offsetting on-the-fly.

    This class is intended to be internal, since SelectionNodeSet instances can be
    freely created but only "apply" for offsetting when registered globally,
    in which case it delegates the processing to PopulationNodes.

    We store internal class-level _global_populations so that offsets are truly
    global, independently of the CellManager
    """

    _global_populations = []
    """Populations which may have offset"""
    _do_offsetting = True
    """False will freeze offsets to ensure final gids are consistent"""

    def __init__(self, name):
        """Ctor for a group of nodes belonging to the same population.
        It wont probably be used publicly given `get()` is also a factory.
        """
        self.name = name
        self.nodesets = WeakList()  # each population might contain several SelectionNodeSet's
        self.max_gid = 0  # maximum raw gid (without offset)
        self.offset = 0

    def _append(self, nodeset):
        self.nodesets.append(nodeset)
        self._update(nodeset)
        return self

    def _update(self, updated_nodeset):
        updated_nodeset._offset = self.offset
        if not self._do_offsetting:
            return
        local_max = max(self.max_gid, updated_nodeset._max_gid)
        max_gid = int(MPI.allreduce(local_max, MPI.MAX))
        if max_gid > self.max_gid:
            self.max_gid = max_gid
            self._update_offsets()

    @classmethod
    def register(cls, population, nodeset, **create_kw):
        return cls.get(population, create=True, **create_kw)._append(nodeset)

    @classmethod
    def freeze_offsets(cls):
        cls._do_offsetting = False

    @classmethod
    def reset(cls):
        cls._global_populations.clear()
        cls._do_offsetting = True

    @classmethod
    def all(cls):
        return cls._global_populations

    @classmethod
    def get(cls, population_name, *, create=False, **create_kw):
        obj = next(filter(lambda x: x.name == population_name, cls._global_populations), None)
        if not obj and create:
            obj = cls.create_pop(population_name, **create_kw)
        return obj

    @classmethod
    def create_pop(cls, population_name):
        new_population = cls(population_name)
        cls._global_populations.append(new_population)
        cls._global_populations = sorted(cls._global_populations, key=lambda x: x.name)
        new_population._compute_offset(cls._find_previous(new_population))
        return new_population

    @classmethod
    def _find_previous(cls, cur_pop):
        prev_nodeset = None
        for obj in cls.all():
            if obj is cur_pop:
                return prev_nodeset
            prev_nodeset = obj
        return None

    def _compute_offset(self, prev_gidpop):
        offset = 0
        # This offset is gonna be the offset+max_gid of the previous population, round up
        if prev_gidpop is not None:
            cur_max = prev_gidpop.offset + prev_gidpop.max_gid
            # round up 1000's. GIDs are 1 based: Blocks [1-1000], [1001-2000]
            offset = ((cur_max - 1) // 1000 + 1) * 1000
        self.offset = offset
        # Update individual nodesets
        for nodeset in self.nodesets:  # nodeset is a weakref
            nodeset()._offset = offset

    def _update_offsets(self):
        """Update all global offsets after adding gids"""
        update = False
        prev_gidpop = None
        for gidpop in self.all():
            if gidpop is self:
                update = True
                prev_gidpop = gidpop
                continue
            if update:
                # We are in a subsequent nodeset - re-set offsetting
                gidpop._compute_offset(prev_gidpop)
            prev_gidpop = gidpop

    @classmethod
    @contextmanager
    def offset_freezer(cls):
        cls._do_offsetting = False
        yield
        cls._do_offsetting = True


class SelectionNodeSet:
    """Set of nodes with optional global registration and offset handling.

    A shim over libsonata.Selection with optional populations, offsets and MEtype metadata per gid
    """

    def __init__(self, gids=None, gid_info=None):
        """Init.

        Args:
            gids: The gids to handle
            gid_info: a map containing METype information about each cell.
                In v5 and v6 values are METypeItem's
        """
        self._offset = 0
        self._max_gid = 0  # maximum raw gid (without offset)
        self._population_group = None  # register in a population so gids can be unique
        self._selection = libsonata.Selection([])  # raw
        self._gid_info = {}
        if isinstance(gids, libsonata.Selection):
            self.add_selection(gids, gid_info)
        else:
            self.add_gids(gids, gid_info)

    offset = property(lambda self: self._offset)
    max_gid = property(lambda self: self._max_gid)

    def __repr__(self):
        gids = self.gids(raw_gids=True)
        n = len(gids)
        return (
            f"SelectionNodeSet(n={n}, "
            f"offset={self.offset}, "
            f"population={self.population_name}, "
            f"raw gids={gids})"
        )

    def __len__(self):
        return self._selection.flat_size

    def __iter__(self):
        for start, stop in self._selection.ranges:
            yield from range(start, stop)

    def iter_cell_info(self, raw_gids=True):
        """Iterate over GIDs with optional offset and metadata"""
        offset_add = 0 if raw_gids else self._offset

        for gid in self:
            yield gid + offset_add, self._gid_info.get(gid)

    def selection(self, raw_gids):
        """Return the internal Selection, optionally offset by the population"""
        if raw_gids:
            return self._selection
        return libsonata.Selection(
            [(start + self._offset, stop + self._offset) for start, stop in self._selection.ranges]
        )

    def gids(self, raw_gids):
        """Return all GIDs as a flat array, optionally offset by the population"""
        return np.asarray(self.selection(raw_gids=raw_gids).flatten(), dtype="uint32")

    def register_global(self, population_name):
        """Register this nodeset in a global population group

        Args:
            population_name: The name of the population these ids belong to
        """
        self._population_group = PopulationNodes.register(population_name, self)
        return self

    @property
    def population_name(self):
        return self._population_group.name if self._population_group else None

    def _check_update_offsets(self):
        """Check/reset offsets based on the other populations"""
        if self._population_group:
            self._population_group._update(self)  # Note: triggers a reduce.

    def add_selection(self, selection: libsonata.Selection, gid_info=None):
        """Add libsonata.Selection GIDs and optional metadata, updating offsets and max_gid

        Args:
            selection: libsonata.Selection of GIDs
            gid_info: Optional map of GID to METype info (v5/v6 values are METypeItem)
        """
        if selection is None:
            return
        self._selection |= selection
        if self:
            # libsonata.Selection.ranges may be unsorted
            # Probably not needed since add_gids sorts
            self._max_gid = max(self.max_gid, np.max([i - 1 for _, i in self._selection.ranges]))
        if gid_info:
            self._gid_info.update(gid_info)
        self._check_update_offsets()  # check offsets (uses reduce)

    def add_gids(self, gids: list[int], gid_info=None):
        """Add GIDs and optional metadata, updating offsets and max_gid

        Args:
            gids: GIDs to add (list)
            gid_info: Optional map of GID to METype info (v5/v6 values are METypeItem)
        """
        if gids is None:
            return
        self.add_selection(selection=libsonata.Selection(gids), gid_info=gid_info)

    def intersection(self, other: SelectionNodeSet, raw_gids=False) -> libsonata.Selection:
        """Return libsonata.Selection in common with another nodeset

        For nodesets to intersect they must belong to the same population and
        have common gids
        """
        if not isinstance(other, SelectionNodeSet):
            raise TypeError(f"Expected SelectionNodeSet, got {type(other).__name__}")
        if self.population_name != other.population_name:
            return libsonata.Selection([])

        ans = self._selection & other._selection
        if raw_gids:
            return ans
        return libsonata.Selection(
            [(start + self.offset, stop + self.offset) for start, stop in ans.ranges]
        )

    def intersects(self, other):
        """Check if the current nodeset intersects another

        For nodesets to intersect they must belong to the same population and
        have common gids
        """
        return bool(self.intersection(other))

    def clear_cell_info(self):
        """Clear all stored GID metadata"""
        self._gid_info = None
