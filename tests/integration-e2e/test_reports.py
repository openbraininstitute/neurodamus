
from pathlib import Path

import libsonata
import numpy.testing as npt
import pytest

from neurodamus.core.configuration import SimConfig


def _read_sonata_report(report_file):
    report = libsonata.ElementReportReader(report_file)
    pop_name = report.get_population_names()[0]
    node_ids = report[pop_name].get_node_ids()
    data = report[pop_name].get()
    return node_ids, data

@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "v5_sonata_config",
        "extra_config": {
            "node_set": "Mosaic",
            "reports": {
                "summation_report": {
                    "type": "summation",
                    "cells": "Mosaic",
                    "variable_name": "i_membrane,IClamp",
                    "sections": "all",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 40.0
                },
                "synapse_report" : {
                    "type": "synapse",
                    "cells": "Mosaic",
                    "variable_name": "ProbAMPANMDA_EMS.g",
                    "sections": "all",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 40.0
                },
                "summation_ProbGABAAB": {
                    "type": "summation",
                    "cells": "Mosaic",
                    "variable_name": "ProbGABAAB_EMS.i",
                    "sections": "all",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 40.0
                }
            }
        }
    }
], indirect=True)
@pytest.mark.slow
def test_v5_sonata_reports(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()

    report_refs = {
        "soma_report.h5":
            [(10, 3, -64.92565), (128, 1, -60.309418), (333, 4, -39.864296)],
        "summation_report.h5":
            [(20, 153, 1.19864846e-4), (60, 42, 1.1587787e-4), (283, 121, 3.3678625e-5)]
    }
    node_id_refs = [0, 1, 2, 3, 4]

    # Go through each report and compare the results
    for report_name, refs in report_refs.items():
        result_ids, result_data = _read_sonata_report(Path(SimConfig.output_root) / report_name)
        assert result_ids == node_id_refs
        for row, col, ref in refs:
            npt.assert_allclose(result_data.data[row][col], ref)
