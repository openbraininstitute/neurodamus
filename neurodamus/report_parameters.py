from dataclasses import dataclass
from typing import Literal
from enum import IntEnum
from pathlib import Path
import logging


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
        if (s is None or s == ""):
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
    collapse_into_soma: bool

def check_report_parameters(rep_params: ReportParameters, Nd_dt: float, lfp_active: bool) -> None:
    """Validate report parameters against simulation constraints."""
    if rep_params.start > rep_params.end:
        raise ReportSetupError(
            f"Invalid report configuration: end time ({rep_params.end}) is before "
            f"start time ({rep_params.start})."
        )

    if rep_params.dt < Nd_dt:
        raise ReportSetupError(
            f"Invalid report configuration: report dt ({rep_params.dt}) is smaller than "
            f"simulation dt ({Nd_dt})."
        )

    if rep_params.type == ReportType.LFP and not lfp_active:
        raise ReportSetupError(
            "LFP report setup failed: electrodes file may be missing "
            "or simulator is not set to CoreNEURON."
        )

def create_report_parameters(duration, Nd_t, output_root, rep_name, rep_conf, target, buffer_size):
    """Create report parameters from configuration."""
    start_time = rep_conf["StartTime"]
    end_time = rep_conf.get("EndTime", duration)
    rep_dt = rep_conf["Dt"]
    rep_type = ReportType.from_string(rep_conf["Type"])
    if Nd_t > 0:
        start_time += Nd_t
        end_time += Nd_t

    sections=SectionType.from_string(rep_conf.get("Sections"))
    compartments=Compartments.from_string(rep_conf.get("Compartments"))
    collapse_into_soma = (sections == SectionType.SOMA) and (compartments == Compartments.CENTER) and rep_type != ReportType.COMPARTMENT_SET
    if rep_type == ReportType.SUMMATION and collapse_into_soma:
        sections = SectionType.ALL
        compartments = Compartments.ALL

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
        collapse_into_soma=collapse_into_soma
    )



    # def _report_build_params(self, rep_name, rep_conf):
    #     """Build and validate report parameters from configuration.

    #     Ensures report timing and settings are consistent with simulation constraints,
    #     raises ReportSetupError on invalid configurations (e.g., missing LFP setup,
    #     invalid time ranges, or incompatible time steps).

    #     Returns:
    #         ReportParams: A populated report parameters object.
    #     """

        

    #     sim_end = self._run_conf["Duration"]
    #     rep_type = rep_conf["Type"]
    #     start_time = rep_conf["StartTime"]
    #     end_time = rep_conf.get("EndTime", sim_end)
    #     rep_dt = rep_conf["Dt"]
    #     rep_format = rep_conf["Format"]

    #     lfp_disabled = not self._circuits.global_manager._lfp_manager._lfp_file
    #     if rep_type == "lfp" and lfp_disabled:
    #         raise ReportSetupError(
    #             "LFP report setup failed: electrodes file may be missing "
    #             "or simulator is not set to CoreNEURON."
    #         )
    #     logging.info(
    #         " * %s (Type: %s, Target: %s, Dt: %f)",
    #         rep_name,
    #         rep_type,
    #         rep_conf["Target"],
    #         rep_dt,
    #     )

    #     if Nd.t > 0:
    #         start_time += Nd.t
    #         end_time += Nd.t
    #     end_time = min(end_time, sim_end)
    #     if start_time > end_time:
    #         raise ReportSetupError(
    #             f"Invalid report configuration: end time ({end_time}) is before "
    #             f"start time ({start_time})."
    #         )

    #     if rep_dt < Nd.dt:
    #         raise ReportSetupError(
    #             f"Invalid report configuration: report dt ({rep_dt}) is smaller than "
    #             f"simulation dt ({Nd.dt})."
    #         )

    #     rep_target = TargetSpec(rep_conf["Target"])
    #     population_name = (
    #         rep_target.population or self._target_spec.population or self.__default___population
    #     )
    #     log_verbose("Report on Population: %s, Target: %s", population_name, rep_target.name)

    #     report_on = rep_conf["ReportOn"]
    #     return ReportParams(
    #         rep_type,  # rep type is case sensitive !!
    #         os.path.basename(rep_conf.get("FileName", rep_name)),
    #         report_on,
    #         rep_conf["Unit"],
    #         rep_format,
    #         rep_dt,
    #         start_time,
    #         end_time,
    #         SimConfig.output_root,
    #         rep_conf.get("Scaling"),
    #     )


    