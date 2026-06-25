import pytest
from pathlib import Path
import libsonata

from ..conftest import RINGTEST_DIR


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "reports": {
                "lfp_report": {
                    "type": "lfp",
                    "cells": "Mosaic",
                    "electrodes_file": str(RINGTEST_DIR / "lfp_file.h5"),
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 2.0
                }
            },
            "inputs": {
                "stimulus_pulse": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 1,
                    "duration": 50,
                    "node_set": "RingA",
                    "represents_physical_electrode": True,
                    "amp_start": 10,
                    "width": 1,
                    "frequency": 50
                }
            }
        }
    },
], indirect=True)
@pytest.mark.forked
def test_lfp_reports(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core.configuration import SimConfig
    from neurodamus.core.coreneuron_configuration import CoreConfig

    nd = Neurodamus(create_tmp_simulation_config_file)

    assert Path(CoreConfig.report_config_file_save).exists()

    assert len(SimConfig.reports) == 1
    rep_name, rep_config = next(iter(SimConfig.reports.items()))
    assert rep_name == 'lfp_report'
    assert rep_config.type == libsonata.SimulationConfig.Report.Type.lfp
    assert rep_config.cells == 'Mosaic'
    assert rep_config.start_time == 0.0
    assert rep_config.end_time == 2.0
    assert rep_config.dt == 0.1
    assert rep_config.variable_name == ''
    assert rep_config.file_name == str(Path(CoreConfig.output_root) / (rep_name + ".h5"))

    nd.run()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
        }
    },
], indirect=True)
def test_missing_electrodes_file(create_tmp_simulation_config_file):
    """Test that a missing electrodes_file does not crash when no LFP reports are configured."""
    from neurodamus import Neurodamus

    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    nd.run()
