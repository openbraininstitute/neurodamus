import logging
import struct

from ._utils import run_only_rank0
from neurodamus.report_parameters import ReportType


class CoreReportConfigEntry:
    # (field_name, type, init input)
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
        assert isinstance(gids, list), "gids must be a list"
        assert len(gids) > 0, "gids cannot be empty"
        self._gids = gids

    @run_only_rank0
    def set_points(self, gids, section_ids, compartment_ids):
        assert self.report_type == "compartment_set", (
            f"set_points is only compatible with 'compartment_set' report type, got {self.report_type}"
        )
        assert len(gids) > 0, "gids cannot be empty"
        assert isinstance(gids, list), f"gids must be a list, got {type(gids).__name__}"
        assert isinstance(section_ids, list), (
            f"section_ids must be a list, got {type(section_ids).__name__}"
        )
        assert isinstance(compartment_ids, list), (
            f"compartment_ids must be a list, got {type(compartment_ids).__name__}"
        )
        assert len(gids) == len(section_ids) == len(compartment_ids), (
            f"gids, section_ids, and compartment_ids must have the same length, got: {len(gids)}, {len(section_ids)}, {len(compartment_ids)}"
        )

        self._gids = gids
        self._points_section_id = section_ids
        self._points_compartment_id = compartment_ids

    @run_only_rank0
    def dump(self, f):
        if not self._gids:
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
        print("line:", line)
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
            report_type=rep_params.type,
            report_on=rep_params.report_on,
            unit=rep_params.unit,
            format=rep_params.format,
            sections=rep_params.sections,
            compartments=rep_params.compartments,
            dt=rep_params.dt,
            start=rep_params.start,
            end=rep_params.end,
            buffer_size=rep_params.buffer_size,
            scaling=rep_params.scaling,
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


class CoreReportConfig:
    """Handler of report.conf, the configuration of the coreNeuron reports. This class manages how the file is handled"""

    def __init__(self):
        # key: report_name, value: CoreReportConfigEntry
        self.reports: dict[str, CoreReportConfigEntry] = {}

    def __eq__(self, other):
        if not isinstance(other, CoreReportConfig):
            return NotImplemented
        return self.reports == other.reports

    def add_entry(self, entry: CoreReportConfigEntry):
        logging.info(
            "Adding report %s for CoreNEURON with %s gids", entry.report_name, entry.num_gids
        )
        self.reports[entry.report_name] = entry

    @run_only_rank0
    def dump(self, path: str):
        # always override
        with open(path, "wb") as f:
            # number of reports
            f.write(f"{len(self.reports)}\n".encode())
            # dump each entry
            for entry in self.reports.values():
                entry.dump(f)

    @classmethod
    def load(cls, path: str) -> "CoreReportConfig":
        config = cls()
        with open(path, "rb") as f:
            line = f.readline()
            if not line:
                return config  # empty file
            try:
                num_reports = int(line.strip())
            except ValueError:
                raise ValueError(f"Invalid number of reports in file: {line}")

            for _ in range(num_reports):
                entry = CoreReportConfigEntry.load_from_file(f)
                if entry is None:
                    raise ValueError("Unexpected EOF while reading reports")
                config.add_entry(entry)

        return config
