
from pathlib import Path

import libsonata
import numpy.testing as npt
import pytest

from neurodamus import Neurodamus
from neurodamus.core.configuration import SimConfig
from ..conftest import V5_SONATA

import numpy as np


def _read_sonata_report(report_file):
    """Return node IDs and data from a SONATA report file."""
    report = libsonata.ElementReportReader(report_file)
    pop_name = report.get_population_names()[0]
    node_ids = report[pop_name].get_node_ids()
    data = report[pop_name].get()
    return node_ids, data

def _sum_data_by_gid(data):
    """ Sum data by gid to mirror a summation report (without scaling) """
    data_np = np.array(data.data) 
    gids = np.array([i[0] for i in data.ids])
    unique_gids, inverse_indices = np.unique(gids, return_inverse=True)
    ans = np.zeros((data_np.shape[0], len(unique_gids)))
    np.add.at(ans, (slice(None), inverse_indices), data_np)
    return ans

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "v5_sonata_config",
#         "extra_config": {
#             "node_set": "Mosaic",
#             "reports": {
#                 "summation_report": {
#                     "type": "summation",
#                     "cells": "Mosaic",
#                     "variable_name": "i_membrane,IClamp",
#                     "sections": "all",
#                     "dt": 0.1,
#                     "start_time": 0.0,
#                     "end_time": 40.0
#                 },
#                 "synapse_report" : {
#                     "type": "synapse",
#                     "cells": "Mosaic",
#                     "variable_name": "ProbAMPANMDA_EMS.g",
#                     "sections": "all",
#                     "dt": 0.1,
#                     "start_time": 0.0,
#                     "end_time": 40.0
#                 },
#                 "summation_ProbGABAAB": {
#                     "type": "summation",
#                     "cells": "Mosaic",
#                     "variable_name": "ProbGABAAB_EMS.i",
#                     "sections": "all",
#                     "dt": 0.1,
#                     "start_time": 0.0,
#                     "end_time": 40.0
#                 },
#                 "compartment_v": {
#                     "type": "compartment",
#                     "sections": "soma",
#                     "compartment": "all",
#                     "variable_name": "v",
#                     "dt": 0.1,
#                     "start_time": 0.0,
#                     "end_time": 40.0
#                 },
#                 "compartment_set_v": {
#                     "type": "compartment_set",
#                     "compartment_set": "cs1",
#                     "variable_name": "v",
#                     "dt": 0.1,
#                     "start_time": 0.0,
#                     "end_time": 40.0
#                 }
#             },
#         }
#     }
# ], indirect=True)
# @pytest.mark.slow
# def test_v5_sonata_reports(create_tmp_simulation_config_file):
#     nd = Neurodamus(create_tmp_simulation_config_file)
#     output_dir = Path(SimConfig.output_root)
#     nd.run()

#     report_refs = {
#         "soma_report.h5": [(10, 3, -64.92565), (128, 1, -60.309418), (333, 4, -39.864296)],
#         "summation_report.h5": [(20, 153, 1.19864846e-4), (60, 42, 1.1587787e-4), (283, 121, 3.3678625e-5)]
#     }
#     node_ids = list(range(5))

#     # Go through each report and compare the results
#     for report_name, refs in report_refs.items():
#         result_ids, result_data = _read_sonata_report(output_dir / report_name)
#         assert result_ids == node_ids
#         res = [result_data.data[row][col] for row, col, _ref in refs]
#         ref = [v for _row, _col, v in refs]
#         npt.assert_allclose(res, ref)
    
#     # test compartment_sets_v.h5
#     node_ids = [0, 2, 3]
#     refs = [(22, 1, -64.14941), (36, 3, -63.708347), (48, 7, -64.82845)]
#     result_ids, result_data = _read_sonata_report(output_dir / "compartment_set_v.h5")
#     assert result_ids == node_ids
#     assert result_data.data.shape[1] == 8
#     res = [result_data.data[row][col] for row, col, _ref in refs]
#     ref = [v for _row, _col, v in refs]
#     npt.assert_allclose(res, ref)

#     # test compare compartment_v.h5 with compartment_set_v.h5
#     _, soma_v_data = _read_sonata_report(output_dir / "compartment_v.h5")
#     for col_res, col_ref in [(0, 0), (4, 2), (6, 3)]:
#         npt.assert_allclose(result_data.data[:,col_res], soma_v_data.data[:,col_ref])

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "inputs": {
                    "Stimulus": {
                        "module": "pulse",
                        "input_type": "current_clamp",
                        "represents_physical_electrode": True,
                        "amp_start": 3,
                        "width": 10,
                        "frequency": 50,
                        "delay": 0,
                        "duration": 50,
                        "node_set": "Mosaic"
                    },
                },
                "target_simulator": "CORENEURON",
                "reports": {
                    "compartment_v": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "v",
                        "sections": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    },
                    "summation_v": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "v",
                        "sections": "soma",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                    "compartment_i_membrane": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "i_membrane",
                        "sections": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    },
                    "summation_i_membrane": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "i_membrane",
                        "sections": "soma",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                    # "compartment_pas": {
                    #     "type": "compartment",
                    #     "cells": "Mosaic",
                    #     "variable_name": "pas",
                    #     "sections": "all",
                    #     "dt": 1,
                    #     "start_time": 0.0,
                    #     "end_time": 40.0,
                    # },
                    "summation_pas": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "pas",
                        "sections": "soma",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                },
            },
        }
    ],
    indirect=True,
)
@pytest.mark.slow
def test_summation_vs_compartment_reports(create_tmp_simulation_config_file):
    """
    Test that the summation report matches the summed compartment report.

    Runs a simulation generating both compartment and summation reports for 'pas',
    then asserts that summing compartment data per gid equals the summation report data,
    within numerical tolerance.
    """
    nd = Neurodamus(create_tmp_simulation_config_file)
    output_dir = Path(SimConfig.output_root)

    nd.run()

    for var in ["v", "i_membrane"]:
        _compartment_ids, compartment_data = _read_sonata_report(output_dir / f"compartment_{var}.h5")

        compartment_data_sum_by_gid = _sum_data_by_gid(compartment_data)
        _summation_ids, summation_data = _read_sonata_report(output_dir / f"summation_{var}.h5")

        assert np.allclose(compartment_data_sum_by_gid[:, :], summation_data.data[:,:], atol=1e-6)
