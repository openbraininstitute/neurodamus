from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import libsonata

from .utils.pyutils import cache_errors

if TYPE_CHECKING:
    from neurodamus.target_manager import TargetPointList


class ReportSetupError(Exception):
    pass


@dataclass
class ReportParameters:
    type: libsonata.SimulationConfig.Report.Type
    name: str
    report_on: str
    unit: str
    format: str
    dt: float
    start: float
    end: float
    output_dir: str
    buffer_size: int
    scaling: libsonata.SimulationConfig.Report.Scaling
    target: object
    sections: libsonata.SimulationConfig.Report.Sections
    compartments: libsonata.SimulationConfig.Report.Compartments
    compartment_set: str
    points: list[TargetPointList] | None = (
        None  # this is filled later with get_point_list
    )


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

    if rep_params.type == libsonata.SimulationConfig.Report.Type.lfp and not lfp_active:
        errors.append(
            "LFP report setup failed: electrodes file may be missing or "
            "simulator is not set to CoreNEURON."
        )

    if errors:
        raise ReportSetupError("\n".join(errors))


@cache_errors
def create_report_parameters(
    sim_end, nd_t, output_root, rep_name, rep_conf, target, buffer_size
):
    """Create report parameters from configuration and CLI"""
    start_time = rep_conf.start_time + max(0, nd_t)
    end_time = min(rep_conf.end_time + max(0, nd_t), sim_end)

    logging.info(
        " * %s (Type: %s, Target: %s, Dt: %f)",
        rep_name,
        rep_conf.type,
        rep_conf.cells,
        rep_conf.dt,
    )

    return ReportParameters(
        type=rep_conf.type,
        name=Path(rep_conf.file_name).name,
        report_on=rep_conf.variable_name,
        unit=rep_conf.unit,
        format="SONATA",
        dt=rep_conf.dt,
        start=start_time,
        end=end_time,
        output_dir=output_root,
        buffer_size=buffer_size,
        scaling=rep_conf.scaling,
        target=target,
        sections=rep_conf.sections,
        compartments=rep_conf.compartments,
        compartment_set=rep_conf.compartment_set,
    )
