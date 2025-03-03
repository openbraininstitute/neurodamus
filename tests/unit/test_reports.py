import pytest
from pathlib import Path

from neurodamus.node import Node


@pytest.mark.slow
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "reports": {
                "new_report": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "wrong",
                    "sections": "all",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 40.0,
                }
            }
        }
    }
], indirect=True)
def test_report_config_error(create_tmp_simulation_config_file):
    with pytest.raises(Exception):
        n = Node(create_tmp_simulation_config_file)
        n.load_targets()
        n.create_cells()
        n.enable_reports()


@pytest.mark.slow
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "reports": {
                "new_report": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "wrong",
                    "sections": "all",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 40.0,
                    "enabled": False
                }
            }
        }
    }
], indirect=True)
def test_report_disabled(create_tmp_simulation_config_file):
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_reports()
    assert len(n.reports) == 0


@pytest.mark.slow
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "reports": {
                "soma_v": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "v",
                    "sections": "soma",
                    "dt": 10,
                    "start_time": 0.0,
                    "end_time": 40.0,
                },
                "compartment_v": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "v",
                    "sections": "all",
                    "dt": 5,
                    "start_time": 0.0,
                    "end_time": 50.0,
                },
            }
        }
    }
], indirect=True)
def test_neuorn_report(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    from neurodamus.core.configuration import SimConfig
    from neurodamus.core import NeurodamusCore as Nd
    reports_conf = {name: conf for name, conf in SimConfig.reports.items() if conf["Enabled"]}
    reports = {}
    for rep_name, rep_conf in reports_conf.items():
        # recorder, tvec = record_report(rep_conf, Nd, n._target_manager)
        reports[rep_name] = (record_report(rep_conf, Nd, n._target_manager))

    Nd.finitialize()   # reinit for the recordings to be registered
    n.run()
    for rep_name, (recorder, tvec) in reports.items():
        write_report(Path(n._run_conf["OutputRoot"]) / (rep_name + ".txt"), recorder, tvec)


def record_report(rep_conf, Nd, target_manager):
    from neurodamus.target_manager import TargetSpec
    sections = rep_conf.get("Sections")
    compartments = rep_conf.get("Compartments")
    rep_type = rep_conf["Type"]
    variable_name = rep_conf["ReportOn"]
    start_time = rep_conf["StartTime"]
    stop_time = rep_conf["EndTime"]
    dt = rep_conf["Dt"]

    tvec = Nd.Vector()
    tvec.indgen(start_time, stop_time, dt)

    target_spec = TargetSpec(rep_conf["Target"])
    target = target_manager.get_target(target_spec)
    sum_currents_into_soma = sections == "soma" and compartments == "center"
    # In case of summation in the soma, we need all points anyway
    if sum_currents_into_soma and rep_type == "Summation":
        sections = "all"
        compartments = "all"
    points = target_manager.getPointList(
        target, sections=sections, compartments=compartments
    )
    recorder = []
    for point in points:
        gid = point.gid
        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            # Enable fast_imem calculation in Neuron
            if variable_name == "i_membrane":
                Nd.cvode.use_fast_imem(1)
                variable_name = "i_membrane_"
            var_ref = getattr(section(x), "_ref_" + variable_name)
            voltage_vec = Nd.Vector()
            voltage_vec.record(var_ref, tvec)
            segname = str(section(x))
            segname = segname[segname.find(".")+1:]
            recorder.append((gid, segname, voltage_vec))
    return recorder, tvec


def write_report(report_name, recorder, tvec):
    with open(report_name, "w") as f:
        f.write(f'{"cell_id":<10}{"seg_name":<20}{"time":<10}{"data":<30}\n')
        for gid, secname, data_vec in recorder:
            for t, data in zip(tvec, data_vec):
                f.write(f"{gid:<10}{secname:<20}{t:<10}{data:<30}\n")
