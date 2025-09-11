from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import get_type_hints

from ._utils import run_only_rank0


# test
@dataclass
class CoreSimulationConfig:
    # required
    outpath: str = field(metadata={"file_key": "outpath", "is_path": True})
    datpath: str = field(metadata={"file_key": "datpath", "is_path": True})
    tstop: float
    dt: float
    prcellgid: int
    celsius: float
    voltage: float
    cell_permute: int = field(metadata={"file_key": "cell-permute"})
    mpi: int

    # optional
    pattern: str = None
    seed: int = None
    model_stats: bool = field(default=None, metadata={"file_key": "model-stats"})
    report_conf: str = field(default=None, metadata={"file_key": "report-conf", "is_path": True})

    def __post_init__(self):
        # type coercion and path resolution
        hints = get_type_hints(self)
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            coerced = self._coerce_type(hints[f.name], val)
            if f.metadata.get("is_path", False) and coerced is not None:
                coerced = str(Path(coerced).resolve())
            setattr(self, f.name, coerced)

    @staticmethod
    def _coerce_type(typ, val):
        result = None

        if val is not None:
            if isinstance(val, typ):
                result = val
            elif typ is bool:
                result = val.strip().lower() in {"true", "1"} if isinstance(val, str) else bool(val)
            elif typ is int:
                result = int(val)
            elif typ is float:
                result = float(val)
            elif typ is str:
                result = str(val)
            else:
                result = val

        return result

    @run_only_rank0
    def dump(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Writing coreneuron simulation config file: %s", path.resolve())

        with path.open("w", encoding="utf-8") as fp:
            for f in fields(self):
                file_key = f.metadata.get("file_key", f.name)
                val = getattr(self, f.name)
                if file_key == "model-stats":
                    if val:
                        fp.write("model-stats\n")
                elif val is not None:
                    if isinstance(val, str):
                        fp.write(f"{file_key}='{val}'\n")
                    else:
                        fp.write(f"{file_key}={val}\n")

        logging.info("Done! coreneuron simulation config was written")

    @classmethod
    def load(cls, path: str | Path) -> CoreSimulationConfig:
        # precompute mapping once
        key_map = {f.metadata.get("file_key", f.name): f.name for f in fields(cls)}

        raw = {}
        with Path(path).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "model-stats":
                    raw[key_map["model-stats"]] = True
                    continue
                if "=" not in line:
                    raise ValueError(f"Malformed line in config: {line}")
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'").strip('"')

                if key not in key_map:
                    raise ValueError(f"Unknown config key: {key}")

                raw[key_map[key]] = val
        return cls(**raw)
