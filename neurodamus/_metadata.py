import importlib.metadata

try:
    __version__ = importlib.metadata.version("neurodamus")
    if __version__ is None:
        __version__ = "devel"
except importlib.metadata.PackageNotFoundError:
    __version__ = "devel"

__author__ = "Blue Brain Project/EPFL"
__copyright__ = "Copyright 2005-2023 Blue Brain Project/EPFL"
