import pytest
from pathlib import Path

from neurodamus.node import Node
from tests.utils import record_compartment_report, write_ascii_report, read_ascii_report


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
                    "end_time": 40.0
                },
                "compartment_i": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "i_membrane",
                    "sections": "all",
                    "dt": 10,
                    "start_time": 0.0,
                    "end_time": 40.0
                }
            }
        }
    }
], indirect=True)
def test_neuorn_compartment_report(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core.configuration import SimConfig
    from neurodamus.core import NeurodamusCore as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    assert len(n.reports) == 2

    # For unit tests, we don't build libsonatareport to create the standard sonata reports,
    # instead we use custom functions to record and write report vectors in ASCII format,
    # but currently only for comparment reports
    reports_conf = {name: conf for name, conf in SimConfig.reports.items() if conf["Enabled"]}
    ascii_recorders = {}
    for rep_name, rep_conf in reports_conf.items():
        rep_type = rep_conf["Type"]
        if rep_type == "compartment":
            ascii_recorders[rep_name] = (record_compartment_report(rep_conf, n._target_manager))
    Nd.finitialize()   # reinit for the recordings to be registered
    n.run()

    # Write ASCII reports
    for rep_name, (recorder, tvec) in ascii_recorders.items():
        ascii_report = Path(n._run_conf["OutputRoot"]) / (rep_name + ".txt")
        write_ascii_report(ascii_report, recorder, tvec)

    # Read ASCII reports
    soma_report = Path(n._run_conf["OutputRoot"]) / ("soma_v.txt")
    assert soma_report.exists()
    data = read_ascii_report(soma_report)
    assert len(data) == 25  # 5 time steps * 5 soma sections

    compartment_report = Path(n._run_conf["OutputRoot"]) / ("compartment_i.txt")
    assert compartment_report.exists()
    data = read_ascii_report(compartment_report)
    assert len(data) == 125  # 5 time steps * 5*5 compartments


@pytest.mark.slow
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "reports": {
                "summation": {
                    "type": "summation",
                    "cells": "Mosaic",
                    "variable_name": "i_membrane, IClamp",
                    "unit": "nA",
                    "dt": 10,
                    "start_time": 0.0,
                    "end_time": 40.0
                }
            }
        }
    }
], indirect=True)
def test_enable_summation_report(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    n = Neurodamus(create_tmp_simulation_config_file)
    assert len(n.reports) == 1
