from __future__ import annotations

import logging
from pathlib import Path

from ._utils import run_only_rank0


def to_snake_case(name: str) -> str:
    """Convert a file key with dashes to snake_case."""
    return name.replace("-", "_")


class CoreSimulationConfig:  # noqa: PLW1641
    # field_name (file key), type, mandatory
    SLOTS = [
        ("outpath", str, True),
        ("datpath", str, True),
        ("tstop", float, True),
        ("dt", float, True),
        ("prcellgid", int, True),
        ("celsius", float, True),
        ("voltage", float, True),
        ("cell-permute", int, True),
        ("pattern", str, False),
        ("seed", int, False),
        ("model-stats", bool, False),
        ("report-conf", str, False),
        ("mpi", int, True),
    ]

    def __init__(self, **kwargs):
        for file_key, typ, mandatory in self.SLOTS:
            attr_name = to_snake_case(file_key)

            if file_key in kwargs:
                val = kwargs[file_key]
            elif attr_name in kwargs:  # allow snake_case keys too
                val = kwargs[attr_name]
            elif mandatory:
                raise ValueError(f"Missing required argument: {file_key}")
            else:
                val = None

            val = self._coerce_type(file_key, typ, val)
            setattr(self, attr_name, val)

    @staticmethod
    def _coerce_type(file_key, typ, val):
        """Coerce to expected type. Useful when we want to load the class from file."""
        if val is None or isinstance(val, typ):
            return val

        if typ is bool:
            if isinstance(val, str):
                return val.strip().lower() in {"true", "1", "yes"}
            return bool(val)
        if typ is int:
            return int(val)
        if typ is float:
            return float(val)
        if typ is str:
            return str(val)

        raise TypeError(f"Cannot coerce {file_key} to {typ}")

    @run_only_rank0
    def dump(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Writing sim config file: %s", path)

        with path.open("w", encoding="utf-8") as fp:
            for file_key, _typ, mandatory in self.SLOTS:
                attr_name = to_snake_case(file_key)
                val = getattr(self, attr_name)
                if file_key == "model-stats" and val:
                    fp.write("'model-stats'\n")
                elif mandatory or val not in {None, False, ""}:
                    if isinstance(val, str):
                        fp.write(f"{file_key}='{val}'\n")
                    else:
                        fp.write(f"{file_key}={val}\n")
        logging.info(" => Dataset written to '%s'", path)

    @classmethod
    def load(cls, path: str | Path) -> CoreSimulationConfig:
        raw = {}
        with Path(path).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "'model-stats'":
                    raw["model-stats"] = True
                    continue
                if "=" not in line:
                    raise ValueError(f"Malformed line in config: {line}")
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Strip quotes for string fields
                val = val.strip("'").strip('"')
                raw[key] = val
        return cls(**raw)

    def __repr__(self):
        items = []
        for file_key, _typ, _ in self.SLOTS:
            attr_name = to_snake_case(file_key)
            val = getattr(self, attr_name, None)
            items.append(f"{file_key}={val!r}")
        return f"{self.__class__.__name__}({', '.join(items)})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__dict__ == other.__dict__
