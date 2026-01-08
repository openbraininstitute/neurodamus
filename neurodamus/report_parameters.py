from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .utils.pyutils import StrEnumBase, cache_errors

if TYPE_CHECKING:
    from neurodamus.target_manager import TargetPointList


class ReportSetupError(Exception):
    pass


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


class CompartmentType(StrEnumBase):
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
    compartments: CompartmentType
    compartment_set: str
    points: list[TargetPointList] | None = None  # this is filled later with get_point_list


@cache_errors
def check_report_parameters(
    rep_params: ReportParameters, nd_dt: float, *, lfp_active: bool
) -> None:
    """Validate report parameters against simulation constraints."""
    errors = []
    if rep_params.start > rep_params.end:
        errors.append(
            f"Invalid report configuration: end time ({rep_params.end}) is "
            f"before start time ({rep_params.start})."
        )

    if rep_params.dt < nd_dt:
        errors.append(
            f"Invalid report configuration: report dt ({rep_params.dt}) is smaller "
            f"than simulation dt ({nd_dt})."
        )

    if rep_params.type == ReportType.LFP and not lfp_active:
        errors.append(
            "LFP report setup failed: electrodes file may be missing or "
            "simulator is not set to CoreNEURON."
        )

    if errors:
        raise ReportSetupError("\n".join(errors))


@cache_errors
def create_report_parameters(sim_end, nd_t, output_root, rep_name, rep_conf, target, buffer_size):
    """Create report parameters from configuration."""
    start_time = rep_conf["StartTime"]
    end_time = rep_conf.get("EndTime", sim_end)
    rep_dt = rep_conf["Dt"]
    rep_type = ReportType.from_string(rep_conf["Type"])
    if nd_t > 0:
        start_time += nd_t
        end_time += nd_t
    end_time = min(end_time, sim_end)

    sections = SectionType.from_string(rep_conf.get("Sections"))
    compartments = CompartmentType.from_string(rep_conf.get("Compartments"))

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
        start=start_time,
        end=end_time,
        output_dir=output_root,
        buffer_size=buffer_size,
        scaling=Scaling.from_string(rep_conf.get("Scaling")),
        target=target,
        sections=sections,
        compartments=compartments,
        compartment_set=rep_conf.get("CompartmentSet"),
    )
