from __future__ import annotations

import logging
import struct
from collections.abc import Iterable
from pathlib import Path

from ._utils import run_only_rank0
from neurodamus.report_parameters import ReportType


class CoreReportConfigEntry:  # noqa: PLW1641
    # (field_name, type, is init input)
    SLOTS = [
        ("report_name", str, True),
        ("target_name", str, True),
        ("report_type", str, True),
        ("report_variable", str, True),  # comma-joined from report_on
        ("unit", str, True),
        ("report_format", str, True),
        ("sections", str, True),  # sections.to_string()
        ("compartments", str, True),  # compartments.to_string()
        ("dt", float, True),
        ("start_time", float, True),
        ("end_time", float, True),
        ("num_gids", int, False),  # computed dynamically
        ("buffer_size", int, True),
        ("scaling", str, True),  # scaling.to_string()
    ]

    def __init__(self, *args, **kwargs):
        inputs = [(name, typ) for name, typ, init_input in self.SLOTS if init_input]

        # fill positional args
        for (name, typ), value in zip(inputs, args):
            assert isinstance(value, typ), (
                f"Expected {name} to be {typ.__name__}, got {type(value).__name__}"
            )
            setattr(self, name, value)

        # fill remaining from kwargs
        for name, typ in inputs[len(args) :]:
            if name not in kwargs:
                raise ValueError(f"Missing required argument: {name}")
            value = kwargs[name]
            assert isinstance(value, typ), (
                f"Expected {name} to be {typ.__name__}, got {type(value).__name__}"
            )
            setattr(self, name, value)

        self._gids: list[int] = []
        self._points_section_id: list[int] = []
        self._points_compartment_id: list[int] = []

    @property
    def num_gids(self):
        return len(self._gids)

    @run_only_rank0
    def set_gids(self, gids):
        assert self.report_type != "compartment_set", (
            f"set_gids is not compatible with 'compartment_set' report type, got {self.report_type}"
        )
        if not isinstance(gids, Iterable):
            raise TypeError(f"gids must be iterable, got {type(gids)}")
        assert len(gids) > 0, "gids cannot be empty"
        self._gids = gids

    @run_only_rank0
    def set_points(self, gids, section_ids, compartment_ids):
        assert self.report_type == "compartment_set", (
            "set_points is only compatible with 'compartment_set' report type, "
            f"got {self.report_type}"
        )
        if not isinstance(gids, Iterable):
            raise TypeError(f"gids must be iterable, got {type(gids)}")
        if not isinstance(section_ids, Iterable):
            raise TypeError(f"section_ids must be iterable, got {type(section_ids)}")
        if not isinstance(compartment_ids, Iterable):
            raise TypeError(f"compartment_ids must be iterable, got {type(compartment_ids)}")
        assert len(gids) > 0, "gids cannot be empty"

        self._gids = gids
        self._points_section_id = section_ids
        self._points_compartment_id = compartment_ids

    @run_only_rank0
    def dump(self, f):
        if not len(self._gids):
            raise ValueError(f"Cannot dump entry: gids not set or empty: `{self._gids}`")

        # text line with num_gids in correct position
        line_values = []
        for name, _, _ in self.SLOTS:
            line_values.append(str(getattr(self, name)))
        f.write((" ".join(line_values) + "\n").encode())

        # binary gids
        f.write(struct.pack(f"{self.num_gids}i", *self._gids))
        f.write(b"\n")  # separator
        if self._points_section_id:
            f.write(struct.pack(f"{len(self._points_section_id)}i", *self._points_section_id))
            f.write(b"\n")  # separator
        if self._points_compartment_id:
            f.write(
                struct.pack(f"{len(self._points_compartment_id)}i", *self._points_compartment_id)
            )
            f.write(b"\n")  # separator

    @staticmethod
    def _get_binary_int_array(f, num_elements):
        data = f.read(num_elements * 4)
        if len(data) != num_elements * 4:
            raise ValueError(f"Expected {num_elements * 4} bytes, got {len(data)}")
        f.readline()
        return list(struct.unpack(f"{num_elements}i", data))

    @classmethod
    def load_from_file(cls, f):
        # read text line
        line = f.readline()
        if not line:
            return None  # EOF

        tokens = line.decode().strip().split()
        if len(tokens) != len(cls.SLOTS):
            raise ValueError(f"Expected {len(cls.SLOTS)} fields, got {len(tokens)}")

        # convert tokens to proper types
        kwargs = {}
        for (name, typ, _), tok in zip(cls.SLOTS, tokens):
            if name == "num_gids":
                num_gids = int(tok)  # store to read binary
            else:
                kwargs[name] = typ(tok)

        entry = cls(**kwargs)

        # read binary gids
        gids = CoreReportConfigEntry._get_binary_int_array(f, num_gids)
        if entry.report_type == "compartment_set":
            section_ids = CoreReportConfigEntry._get_binary_int_array(f, num_gids)
            compartment_ids = CoreReportConfigEntry._get_binary_int_array(f, num_gids)
            entry.set_points(gids, section_ids, compartment_ids)
        else:
            entry.set_gids(gids)

        return entry

    @classmethod
    def from_report_params(cls, rep_params):
        entry = cls(
            report_name=rep_params.name,
            target_name=rep_params.target.name,
            report_type=rep_params.type.to_string(),
            report_variable=",".join(rep_params.report_on.split()),
            unit=rep_params.unit,
            report_format=rep_params.format,
            sections=rep_params.sections.to_string(),
            compartments=rep_params.compartments.to_string(),
            dt=rep_params.dt,
            start_time=rep_params.start,
            end_time=rep_params.end,
            buffer_size=rep_params.buffer_size,
            scaling=rep_params.scaling.to_string(),
        )
        if rep_params.type == ReportType.COMPARTMENT_SET:
            # flatten the points for binary encoding
            gids = [i.gid for i in rep_params.points for _section_id, _sec, _x in i]
            points_section_id = [
                section_id for i in rep_params.points for section_id, _sec, _x in i
            ]
            points_compartment_id = [
                sec.sec(x).node_index() for i in rep_params.points for _section_id, sec, x in i
            ]
            entry.set_points(gids, points_section_id, points_compartment_id)
        else:
            entry.set_gids(rep_params.target.get_gids())

        return entry

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self):
        vals = {
            name: getattr(self, name) if name != "num_gids" else self.num_gids
            for name, _, _ in self.SLOTS
        }
        vals["gids"] = self._gids
        vals["points_section_id"] = self._points_section_id
        vals["points_compartment_id"] = self._points_compartment_id
        return f"CoreReportConfigEntry({vals})"


class CoreReportConfig:  # noqa: PLW1641
    """Handler of report.conf, the configuration of the coreNeuron reports.
    This class manages how the file is handled
    """

    __slots__ = ("_pop_offsets", "_reports", "_spike_filename")

    def __init__(self):
        # key: report_name, value: CoreReportConfigEntry
        self._reports: dict[str, CoreReportConfigEntry] = {}
        self._pop_offsets: dict[str, int] | None = None
        self._spike_filename: str | None = None

    def __eq__(self, other):
        if not isinstance(other, CoreReportConfig):
            return NotImplemented
        return (
            self._reports == other._reports
            and self._pop_offsets == other._pop_offsets
            and self._spike_filename == other._spike_filename
        )

    def add_entry(self, entry: CoreReportConfigEntry):
        logging.info(
            "Adding report %s for CoreNEURON with %s gids", entry.report_name, entry.num_gids
        )
        self._reports[entry.report_name] = entry

    def set_pop_offsets(self, pop_offsets: dict[str, int]):
        """Set the population offsets for the reports (skip None key)."""
        assert isinstance(pop_offsets, dict), "pop_offsets must be a dictionary"
        self._pop_offsets = {k: v for k, v in pop_offsets.items() if k is not None}

    def set_spike_filename(self, spike_path: str):
        """Set the spike filename for the reports."""
        if spike_path is not None:
            # Get only the spike file name
            self._spike_filename = spike_path.rsplit("/", maxsplit=1)[-1]

    @run_only_rank0
    def dump(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # always override
        with open(path, "wb") as f:
            # number of reports
            f.write(f"{len(self._reports)}\n".encode())
            # dump each entry
            for entry in self._reports.values():
                entry.dump(f)

            f.write(f"{len(self._pop_offsets)}\n".encode())
            for k, v in self._pop_offsets.items():
                if v is not None:
                    f.write(f"{k} {v}\n".encode())
                else:
                    f.write(f"{k}\n".encode())

            if self._spike_filename:
                f.write(f"{self._spike_filename}\n".encode())

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
        """Update the report configuration file with new substitutions."""
        conf = CoreReportConfig.load(file_path)
        for report_name, targets in substitutions.items():
            report = conf._reports[report_name]
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
