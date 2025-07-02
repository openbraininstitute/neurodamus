"""Internal module which defines several wrapper classes for Hoc entities.
They are then available as singletons objects in neurodamus.core package
"""

import logging
import os
from contextlib import contextmanager

from .configuration import GlobalConfig, SimConfig


# Singleton, instantiated right below
class _Neuron:
    """A wrapper over the neuron simulator."""

    __name__ = "_Neuron"
    _h = None  # We don't import it at module-level to avoid starting neuron
    _hocs_loaded = set()

    # No new attributes. __setattr__ can rely on it
    __slots__ = ()

    @property
    def h(self):
        """The neuron hoc interpreter, initializing if needed."""
        return self._h or self._init()

    @classmethod
    def _init(cls, mpi=False):
        """Initializes the Neuron simulator."""
        if cls._h is not None:
            return cls._h
        if mpi:
            GlobalConfig.set_mpi()

        from neuron import h, nrn

        cls.__cache = {}
        cls._h = h
        cls.Section = nrn.Section
        cls.Segment = nrn.Segment
        h.load_file("stdrun.hoc")
        h("objref nil")
        h.init()
        return h

    @classmethod
    def load_hoc(cls, mod_name):
        """Loads a hoc module, available in the path."""
        if mod_name in cls._hocs_loaded:
            return
        h = cls._h or cls._init()
        mod_filename = mod_name + ".hoc"
        if not h.load_file(mod_filename):
            msg = (
                f"Can't load HOC library {mod_filename}. "
                f"Consider checking HOC_LIBRARY_PATH: `{os.environ.get('HOC_LIBRARY_PATH')}`"
            )
            raise RuntimeError(msg)
        cls._hocs_loaded.add(mod_name)

    @classmethod
    def require(cls, *hoc_mods):
        """Load a set of hoc mods by name."""
        for mod in hoc_mods:
            cls.load_hoc(mod)
        return cls._h

    @classmethod
    def load_dll(cls, dll_path):
        """Loads a Neuron mod file (typically an .so file in linux)."""
        h = cls._h or cls._init()
        rc = h.nrn_load_dll(dll_path)
        if rc == 0:
            raise RuntimeError(
                f"Cant load MOD dll {dll_path}. Please check LD path and dependencies"
            )

    @contextmanager
    def section_in_stack(self, sec):
        """A contect manager to push and pop a section to the Neuron stack."""
        sec.push()
        yield
        self.h.pop_section()

    # Properties that are not found here are get / set
    # directly in neuron.h
    def __getattr__(self, item):
        # We use a cache since going down to hoc costs at least 10us
        # Cache is not expected to grow very large. Unbounded for the moment
        if item.startswith("__"):
            return object.__getattribute__(self, item)
        self._h or self._init()
        cache = self.__class__.__cache
        obj = cache.get(item)
        if obj is None:
            obj = getattr(self._h, item)
            if type(obj) is not float:
                cache[item] = obj
        return obj

    def __setattr__(self, key, value):
        try:
            object.__setattr__(self, key, value)
        except AttributeError:
            setattr(self.h, key, value)


Neuron = _Neuron()
"""A singleton wrapper for the Neuron library"""


class MComplexLoadBalancer:
    """Wrapper of the load balance Hoc Module with mcomplex."""

    def __init__(self, force_regenerate=False):
        # Can we use an existing mcomplex.dat?
        if force_regenerate or not os.path.isfile("mcomplex.dat"):
            logging.info("Generating mcomplex.dat...")
            self._create_mcomplex()
        else:
            logging.info("Using existing mcomplex.dat")
        self._lb = Neuron.h.LoadBalance()
        self._lb.read_mcomplex()

    @staticmethod
    def _create_mcomplex():
        # Save the dt of the simulation and set the dt for calculating the ExperimentalMechComplex
        # to the default value of 0.025
        prev_dt = Neuron.h.dt
        Neuron.h.dt = SimConfig.default_neuron_dt
        # ExperimentalMechComplex is a complex routine changing many state vars, and cant be reused
        # Therefore here we create a temporary LoadBalance obj
        lb = Neuron.h.LoadBalance()
        lb.ExperimentalMechComplex("StdpWA", "extracel", "HDF5", "Report", "Memory", "ASCII")
        # Revert dt to the old value
        Neuron.h.dt = prev_dt
        # mcomplex changes neuron state and results get different. We re-init
        Neuron.h.init()

    def __getattr__(self, item):
        return getattr(self._lb, item)
