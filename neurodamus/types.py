from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EdgePopulationQualified:
    path: Path
    population: str
