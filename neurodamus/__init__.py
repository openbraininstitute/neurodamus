"""neurodamus

The neurodamus package implements the instantiation of simulations in Neuron
based on a configuration file, a.k.a. simulation_config.json
It is deeply based on the HOC implementation, therefore providing python modules like
`node`, `cell_distributor`, etc; and still depends on several low-level HOC files at runtime.
"""

from ._metadata import __version__
from .node import Neurodamus, Node

__all__ = ["Neurodamus", "Node", "__version__"]
