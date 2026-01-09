"""neurodamus

The neurodamus package implements the instantiation of simulations in Neuron
based on a configuration file, a.k.a. simulation_config.json.
"""

from ._metadata import __version__
from .node import Neurodamus, Node

__all__ = ["Neurodamus", "Node", "__version__"]
