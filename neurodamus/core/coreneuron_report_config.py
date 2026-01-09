from __future__ import annotations

import logging
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

import libsonata

from ._utils import run_only_rank0


@dataclass
class CoreReportConfigEntry:
    report_name: str
    target_name: str
    report_type: str
    report_variable: str
    unit: str
    report_format: str
    sections: str
    compartments: str
    dt: float
    start_time: float
    end_time: float
    buffer_size: int
    scaling: str

    # public lists
    gids: list[int] = field(default_factory=list, init=False)
    points_section_ids: list[int] = field(default_factory=list, init=False)
    points_compartment_ids: list[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        # type coercion and path resolution
        hints = get_type_hints(self)
        for f in fields(self):
            val = getattr(self, f.name)
            coerced = self._coerce_type(hints[f.name], val)
            setattr(self, f.name, coerced)

    @staticmethod
    def _coerce_type(typ, val):
        result = None

        typ = get_origin(typ) or typ
        if typ is Union:
            typ = next(t for t in get_args(typ) if t is not type(None))

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

    @property
    def num_gids(self):
        return len(self.gids)

    @run_only_rank0
    def set_gids(self, gids):
        assert self.report_type != "compartment_set"
        assert isinstance(gids, Iterable)
        assert len(gids)
        self.gids = gids

    @run_only_rank0
    def set_points(self, gids, section_ids, compartment_ids):
        assert self.report_type == "compartment_set"
        assert isinstance(gids, Iterable)
        assert isinstance(section_ids, Iterable)
        assert isinstance(compartment_ids, Iterable)
        assert len(gids)
        assert len(gids) == len(section_ids) == len(compartment_ids), (
            f"All input lists must have the same length, "
            f"got gids: {len(gids)}, section_ids: {len(section_ids)}, "
            f"compartment_ids: {len(compartment_ids)}"
        )
        self.gids = gids
        self.points_section_ids = section_ids
        self.points_compartment_ids = compartment_ids

    @run_only_rank0
    def dump(self, f):
        assert len(self.gids)

        # text line with field values
        line_values = [str(getattr(self, f.name)) for f in fields(self) if f.init]
        line_values.insert(11, str(self.num_gids))
        f.write((" ".join(line_values) + "\n").encode())

        # binary gids
        f.write(struct.pack(f"{len(self.gids)}i", *self.gids))
        f.write(b"\n")

        # binary points
        if self.points_section_ids:
            f.write(struct.pack(f"{len(self.points_section_ids)}i", *self.points_section_ids))
            f.write(b"\n")
        if self.points_compartment_ids:
            f.write(
                struct.pack(f"{len(self.points_compartment_ids)}i", *self.points_compartment_ids)
            )
            f.write(b"\n")

    @staticmethod
    def _get_binary_int_array(f, num_elements):
        data = f.read(num_elements * 4)
        if len(data) != num_elements * 4:
            raise ValueError(f"Expected {num_elements * 4} bytes, got {len(data)}")
        f.readline()
        return list(struct.unpack(f"{num_elements}i", data))

    @classmethod
    def load_from_file(cls, f):
        line = f.readline()
        if not line:
            return None
        tokens = line.decode().strip().split()
        num_gids = int(tokens.pop(11))
        entry = cls(*tokens)
        gids = cls._get_binary_int_array(f, num_gids)
        if entry.report_type == "compartment_set":
            section_ids = cls._get_binary_int_array(f, num_gids)
            compartment_ids = cls._get_binary_int_array(f, num_gids)
            entry.set_points(gids, section_ids, compartment_ids)
        else:
            entry.set_gids(gids)
        return entry

    @classmethod
    def from_report_params(cls, rep_params):
        entry = cls(
            report_name=rep_params.name,
            target_name=rep_params.target.name,
            report_type=rep_params.type.name,
            report_variable=",".join(rep_params.report_on.split()),
            unit=rep_params.unit,
            report_format=rep_params.format,
            sections=rep_params.sections.name,
            compartments=rep_params.compartments.name,
            dt=rep_params.dt,
            start_time=rep_params.start,
            end_time=rep_params.end,
            buffer_size=rep_params.buffer_size,
            scaling=rep_params.scaling.name,
        )
        if rep_params.type == libsonata.SimulationConfig.Report.Type.compartment_set:
            gids = [i.gid for i in rep_params.points for _section_id, _sec, _x in i]
            section_ids = [section_id for i in rep_params.points for section_id, _sec, _x in i]
            compartment_ids = [
                sec.sec(x).node_index() for i in rep_params.points for _section_id, sec, x in i
            ]
            entry.set_points(gids, section_ids, compartment_ids)
        else:
            entry.set_gids(rep_params.target.gids(raw_gids=False))
        return entry


@dataclass
class CoreReportConfig:
    """Handler of report.conf, the configuration of the coreNeuron reports."""

    reports: dict[str, CoreReportConfigEntry] = field(default_factory=dict, init=False)
    pop_offsets: dict[str, int] = field(default=None, init=False)
    spike_filename: str = field(default=None, init=False)

    def add_entry(self, entry: CoreReportConfigEntry):
        logging.info(
            "Adding report %s for CoreNEURON with %s gids", entry.report_name, entry.num_gids
        )
        self.reports[entry.report_name] = entry

    def set_pop_offsets(self, pop_offsets: dict[str, int]):
        """Set the population offsets for the reports (skip None key)."""
        assert isinstance(pop_offsets, dict), "pop_offsets must be a dictionary"
        self.pop_offsets = {k: v for k, v in pop_offsets.items() if k is not None}

    def set_spike_filename(self, spike_path: str):
        """Set the spike filename for the reports."""
        if spike_path is not None:
            # Get only the spike file name
            self.spike_filename = spike_path.rsplit("/", maxsplit=1)[-1]

    @run_only_rank0
    def dump(self, path: str | Path):
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # always override
        logging.info("Writing coreneuron report config file: %s", path)
        with open(path, "wb") as f:
            # number of reports
            f.write(f"{len(self.reports)}\n".encode())
            # dump each entry
            for entry in self.reports.values():
                entry.dump(f)

            f.write(f"{len(self.pop_offsets)}\n".encode())
            for k, v in self.pop_offsets.items():
                if v is not None:
                    f.write(f"{k} {v}\n".encode())
                else:
                    f.write(f"{k}\n".encode())

            if self.spike_filename:
                f.write(f"{self.spike_filename}\n".encode())
        logging.info("Done! coreneuron report config was written")

    @classmethod
    def load(cls, path: str) -> CoreReportConfig:
        config = cls()
        with open(path, "rb") as f:
            # --- Read number of reports ---
            line = f.readline()
            if not line:
                return config  # empty file
            try:
                num_reports = int(line.decode().strip())
            except ValueError as err:
                raise ValueError(f"Invalid number of reports in file: {line}") from err

            # --- Read each report ---
            for _ in range(num_reports):
                entry = CoreReportConfigEntry.load_from_file(f)
                if entry is None:
                    raise ValueError("Unexpected EOF while reading reports")
                config.add_entry(entry)

            # --- Mandatory pop_offsets ---
            line = f.readline()
            if not line:
                raise ValueError("Missing population offsets")
            pop_count = int(line.decode().strip())
            pop_offsets = {}
            for _ in range(pop_count):
                line = f.readline()
                if not line:
                    break
                parts = line.decode().strip().split()
                k = parts[0]
                v = int(parts[1]) if len(parts) > 1 else None
                pop_offsets[k] = v
            config.set_pop_offsets(pop_offsets)

            # --- Optional spike filename ---
            line = f.readline()
            if line:
                spike_filename = line.decode().strip()
                if spike_filename:
                    config.set_spike_filename(spike_filename)

        return config

    @staticmethod
    @run_only_rank0
    def update_file(file_path: str, substitutions: dict[str, dict[str, int]]):
        """Update a report configuration file by applying attribute substitutions to
        one or more reports.

        Args:
            file_path (str): Path to the configuration file to load and update.
            substitutions (dict[str, dict[str, int]]): A mapping from report names
                to dictionaries of attribute-value pairs to be updated.

        Example:
            >>> substitutions = {"r1": {"buffer_size": 11}}
            >>> CoreReportConfig.update_file("config.yaml", substitutions)
            # Updates report 'r1' so that r1.buffer_size == 11
        """
        conf = CoreReportConfig.load(file_path)
        for report_name, targets in substitutions.items():
            report = conf.reports[report_name]
            for attr, new_val in targets.items():
                if not hasattr(report, attr):
                    raise AttributeError(f"Missing attribute '{attr}' in {report!r}")

                current_val = getattr(report, attr)
                if not isinstance(new_val, type(current_val)):
                    raise TypeError(
                        f"Type mismatch for '{attr}': expected {type(current_val).__name__}, "
                        f"got {type(new_val).__name__}. "
                        f"Current value={current_val!r}, attempted new value={new_val!r}"
                    )

                setattr(report, attr, new_val)
        conf.dump(file_path)
