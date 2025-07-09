
from pathlib import Path

import libsonata
import numpy.testing as npt
import pytest

from neurodamus import Neurodamus
from neurodamus.core.configuration import SimConfig


def _read_sonata_report(report_file):
    report = libsonata.ElementReportReader(report_file)
    pop_name = report.get_population_names()[0]
    node_ids = report[pop_name].get_node_ids()
    data = report[pop_name].get()
    return node_ids, data

@pytest.mark.slow
@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "inputs": {
                    "Stimulus": {
                        "module": "pulse",
                        "input_type": "current_clamp",
                        "delay": 5,
                        "duration": 50,
                        "node_set": "RingA",
                        "represents_physical_electrode": True,
                        "amp_start": 10,
                        "width": 1,
                        "frequency": 50,
                    }
                },
                "reports": {
                    "bau": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "pas",
                        "sections": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_removeme(create_tmp_simulation_config_file):
    nd = Neurodamus(create_tmp_simulation_config_file)
    output_dir = Path(SimConfig.output_root)
    nd.run()

