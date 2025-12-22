from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QualifiedNodePopulation:
    path: Path
    name: str


@dataclass(frozen=True)
class QualifiedEdgePopulation:
    path: Path
    name: str


NRNPath = QualifiedEdgePopulation | bool

# CellLibraryFile
