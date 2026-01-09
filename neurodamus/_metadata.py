import importlib.metadata

try:
    __version__ = importlib.metadata.version("neurodamus")
except importlib.metadata.PackageNotFoundError:
    __version__ = "devel"

__author__ = "Fernando Pereira <fernando.pereira@epfl.ch>"
__copyright__ = "2018 Blue Brain Project, EPFL"
