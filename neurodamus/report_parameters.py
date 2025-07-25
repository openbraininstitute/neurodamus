import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class ReportSetupError(Exception):
    pass


class StrEnumBase(IntEnum):
    __mapping__: list[tuple[str, int]] = []
    # default for when there is value. Leaving None throws an error
    __default__ = None
    # default when the string is not found in the mapping
    __invalid__ = None

    @classmethod
    def from_string(cls, s: str):
        if not s:
            return cls(cls.__default__)
        mapping = dict(cls.__mapping__)
        return cls(mapping.get(s.lower(), cls.__invalid__))

    def to_string(self) -> str:
        reverse__mapping__ = {v: k for k, v in self.__mapping__}
        return reverse__mapping__[self]

    def __str__(self):
        return f"{self.__class__.__name__}.{self.name}"


class SectionType(StrEnumBase):
    ALL = 0
    SOMA = 1
    AXON = 2
    DEND = 3
    APIC = 4
    INVALID = 5

    __mapping__ = [
        ("all", ALL),
        ("soma", SOMA),
        ("axon", AXON),
        ("dend", DEND),
        ("apic", APIC),
        ("invalid", INVALID),
    ]
    __default__ = SOMA
    __invalid__ = INVALID


class Compartments(StrEnumBase):
    ALL = 0
    CENTER = 1
    INVALID = 2

    __mapping__ = [
        ("all", ALL),
        ("center", CENTER),
        ("invalid", INVALID),
    ]
    __default__ = CENTER
    __invalid__ = INVALID


class Scaling(StrEnumBase):
    NONE = 0
    AREA = 1

    __mapping__ = [
        ("none", NONE),
        ("area", AREA),
    ]
    __default__ = AREA


class ReportType(StrEnumBase):
    COMPARTMENT = 0
    COMPARTMENT_SET = 1
    SUMMATION = 2
    SYNAPSE = 3
    LFP = 4

    __mapping__ = [
        ("compartment", COMPARTMENT),
        ("compartment_set", COMPARTMENT_SET),
        ("summation", SUMMATION),
        ("synapse", SYNAPSE),
        ("lfp", LFP),
    ]


@dataclass
class ReportParameters:
    type: ReportType
    name: str
    report_on: str
    unit: str
    format: str
    dt: float
    start: float
    end: float
    output_dir: str
    buffer_size: int
    scaling: Scaling
    target: object
    sections: SectionType
    compartments: Compartments
    compartment_set: str


def check_report_parameters(
    rep_params: ReportParameters, nd_dt: float, *, lfp_active: bool
) -> None:
    """Validate report parameters against simulation constraints."""
    if rep_params.start > rep_params.end:
        raise ReportSetupError(
            f"Invalid report configuration: end time ({rep_params.end}) is before "
            f"start time ({rep_params.start})."
        )

    if rep_params.dt < nd_dt:
        raise ReportSetupError(
            f"Invalid report configuration: report dt ({rep_params.dt}) is smaller than "
            f"simulation dt ({nd_dt})."
        )

    if rep_params.type == ReportType.LFP and not lfp_active:
        raise ReportSetupError(
            "LFP report setup failed: electrodes file may be missing "
            "or simulator is not set to CoreNEURON."
        )


def create_report_parameters(duration, nd_t, output_root, rep_name, rep_conf, target, buffer_size):
    """Create report parameters from configuration."""
    start_time = rep_conf["StartTime"]
    end_time = rep_conf.get("EndTime", duration)
    rep_dt = rep_conf["Dt"]
    rep_type = ReportType.from_string(rep_conf["Type"])
    if nd_t > 0:
        start_time += nd_t
        end_time += nd_t

    sections = SectionType.from_string(rep_conf.get("Sections"))
    compartments = Compartments.from_string(rep_conf.get("Compartments"))

    logging.info(
        " * %s (Type: %s, Target: %s, Dt: %f)",
        rep_name,
        rep_type,
        rep_conf["Target"],
        rep_dt,
    )

    return ReportParameters(
        type=rep_type,
        name=Path(rep_conf.get("FileName", rep_name)).name,
        report_on=rep_conf["ReportOn"],
        unit=rep_conf["Unit"],
        format=rep_conf["Format"],
        dt=rep_dt,
        start=rep_conf["StartTime"],
        end=end_time,
        output_dir=output_root,
        buffer_size=buffer_size,
        scaling=Scaling.from_string(rep_conf.get("Scaling")),
        target=target,
        sections=sections,
        compartments=compartments,
        compartment_set=rep_conf.get("CompartmentSet"),
    )
