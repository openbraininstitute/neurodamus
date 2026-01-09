"""neurodamus

The neurodamus package implements the instantiation of simulations in Neuron
based on a configuration file, a.k.a. simulation_config.json
It is deeply based on the HOC implementation, therefore providing python modules like
`node`, `cell_distributor`, etc; and still depends on several low-level HOC files at runtime.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "devel"

__author__ = "Fernando Pereira <fernando.pereira@epfl.ch>"
__copyright__ = "2018 Blue Brain Project, EPFL"

from .node import Neurodamus, Node

__all__ = ["Neurodamus", "Node"]
