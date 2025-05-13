import logging
from pathlib import Path

import pytest

from tests import utils
from tests.conftest import RINGTEST_DIR

from neurodamus.core.configuration import SimConfig
from neurodamus.io.sonata_config import SonataConfig
from neurodamus.utils.logging import setup_logging


def test_parse_base():
    raw_conf = SonataConfig(str(RINGTEST_DIR / "simulation_config.json"))
    assert raw_conf._sim_conf.run.random_seed == 1122
    assert raw_conf.parsedRun["BaseSeed"] == 1122


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "conditions": {
                "extracellular_calcium": 1.2,
            }
        }
    },
], indirect=True)
def test_parse_run(create_tmp_simulation_config_file):
    SimConfig.init(create_tmp_simulation_config_file, {})
    # RNGSettings in hoc correctly initialized from Sonata
    assert SimConfig.rng_info.getGlobalSeed() == 1122

    expected_conf = {
        "CircuitTarget": "Mosaic",
        "Simulator": "NEURON",
        "Duration": 50.0,
        "Dt": 0.1,
        "Celsius": 35,
        "V_Init": -65,
        "SpikeLocation": "soma",
        "ExtracellularCalcium": 1.2
    }

    utils.check_is_subset(SimConfig.run_conf, expected_conf)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "conditions": {
                "extracellular_calcium": 1.2,
                "mechanisms": {
                    "GluSynapse": {
                        "cao_CR": 2.0,
                    }
                }
            }
        }
    },
], indirect=True)
def test_extracellular_calcium_warning(create_tmp_simulation_config_file, caplog):
    setup_logging.logging_initted = True
    with caplog.at_level(logging.WARNING):
        SimConfig.init(create_tmp_simulation_config_file, {})
        assert ("Value of cao_CR_GluSynapse (2.0) is not the same as extracellular_calcium (1.2)"
                in caplog.text)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "conditions": {
                "spike_location": "AIS",
                "mechanisms": {
                    "ProbAMPANMDA_EMS": {
                        "init_depleted": False,
                        "minis_single_vesicle": True
                    }
                }
            }
        }
    },
], indirect=True)
def test_parse_conditions(create_tmp_simulation_config_file):
    SimConfig.init(create_tmp_simulation_config_file, {})

    expected_conditions = {
        "init_depleted_ProbAMPANMDA_EMS": False,
        "minis_single_vesicle_ProbAMPANMDA_EMS": True,
        "randomize_Gaba_risetime": "False"
    }

    utils.check_is_subset(next(iter(SimConfig._simulation_config.Conditions.values())),
                          expected_conditions)
    assert SimConfig.run_conf["SpikeLocation"] == "AIS"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "run": {
                "random_seed": 12345,
                "stimulus_seed": 1122
            },
        }
    }
], indirect=True)
def test_parse_seeds(create_tmp_simulation_config_file):
    SimConfig.init(create_tmp_simulation_config_file, {})
    assert SimConfig.rng_info.getGlobalSeed() == 12345
    assert SimConfig.rng_info.getStimulusSeed() == 1122
    assert SimConfig.rng_info.getIonChannelSeed() == 0
    assert SimConfig.rng_info.getMinisSeed() == 0
    assert SimConfig.rng_info.getSynapseSeed() == 0


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "conditions": {
                "modifications": [
                    {
                        "name": "no_SK_E2",
                        "node_set": "single",
                        "type": "ConfigureAllSections",
                        "section_configure": "%%s.gSK_E2bar_SK_E2 = 0"
                    },
                    {
                        "name": "applyTTX",
                        "node_set": "single",
                        "type": "TTX"
                    }
                ]
            }
        }
    }
], indirect=True)
def test_parse_modifications(create_tmp_simulation_config_file):
    SimConfig.init(create_tmp_simulation_config_file, {})
    assert list(SimConfig.modifications.keys()) == ["no_SK_E2", "applyTTX"]  # order preserved
    TTX_mod = SimConfig.modifications["applyTTX"]
    assert TTX_mod["Type"] == "TTX"
    assert TTX_mod["Target"] == "single"
    ConfigureAllSections_mod = SimConfig.modifications["no_SK_E2"]
    ConfigureAllSections_mod["Type"] = "ConfigureAllSections"
    ConfigureAllSections_mod["Target"] = "single"
    ConfigureAllSections_mod["SectionConfigure"] = "%s.gSK_E2bar_SK_E2 = 0"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "connection_overrides": [
                {
                    "name": "GABAB_erev",
                    "source": "Inhibitory",
                    "target": "Mosaic",
                    "weight": 1.0,
                    "synapse_delay_override": 0.5,
                    "synapse_configure": "%s.e_GABAA = -82.0 tau_d_GABAB_ProbGABAAB_EMS = 77",
                    "neuromodulation_dtc": 100,
                    "neuromodulation_strength": 0.75
                }
            ],
        }
    }
], indirect=True)
def test_parse_connections(create_tmp_simulation_config_file):
    SimConfig.init(create_tmp_simulation_config_file, {})
    conn = SimConfig.connections["GABAB_erev"]
    expected_conn = {
        "Source": "Inhibitory",
        "Destination": "Mosaic",
        "Weight": 1.0,
        "Delay": 0,
        "SynDelayOverride": 0.5,
        "SynapseConfigure": "%s.e_GABAA = -82.0 tau_d_GABAB_ProbGABAAB_EMS = 77",
        "NeuromodDtc": 100,
        "NeuromodStrength": 0.75
    }

    utils.check_is_subset(conn, expected_conn)
    assert conn.get("SpontMins") is None
    assert conn.get("Modoverride") is None


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "output": {
                "output_dir": ".",
                "spikes_file": "spikes.h5",
                "spikes_sort_order": "by_time"
            },
        }
    }
], indirect=True)
def test_parse_ouput(create_tmp_simulation_config_file):
    SimConfig.init(create_tmp_simulation_config_file, {})
    # output section
    assert SimConfig.run_conf["SpikesFile"] == "spikes.h5"
    assert SimConfig.run_conf["SpikesSortOrder"] == "by_time"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "inputs": {
                "hypamp_mosaic": {
                    "node_set": "200_L5_PCs",
                    "input_type": "current_clamp",
                    "module": "hyperpolarizing",
                    "delay": 0.0,
                    "duration": 10000.0
                },
                "RelativeShotNoise_L5E_inject": {
                    "node_set": "200_L5_PCs",
                    "input_type": "current_clamp",
                    "module": "relative_shot_noise",
                    "delay": 0.0,
                    "duration": 1000.0,
                    "decay_time": 4.0,
                    "rise_time": 0.4,
                    "relative_skew": 0.63,
                    "mean_percent": 70.0,
                    "sd_percent": 40.0
                },
                "subthreshould_mosaic": {
                    "module": "subthreshold",
                    "input_type": "current_clamp",
                    "delay": 0.0,
                    "duration": 30000.0,
                    "node_set": "Mosaic",
                    "percent_less": 50.0
                }
            }
        }
    }
], indirect=True)
def test_parse_inputs(create_tmp_simulation_config_file):
    SimConfig.init(create_tmp_simulation_config_file, {})

    expected_input_hp = {
        "Pattern": "Hyperpolarizing",
        "Mode": "Current",
        "Target": "200_L5_PCs",
        "Delay": 0.,
        "Duration": 10000.0
    }
    utils.check_is_subset(SimConfig.stimuli[0], expected_input_hp)

    input_RSN = SimConfig.stimuli[1]
    expected_input_RSN = {
        "Pattern": "RelativeShotNoise",
        "Mode": "Current",
        "DecayTime": 4.,
        "RiseTime": 0.4,
        "Delay": 0.,
        "Duration": 1000.,
        "MeanPercent": 70.,
        "RelativeSkew": 0.63,
        "SDPercent": 40.,
        "Dt": 0.25
    }
    utils.check_is_subset(input_RSN, expected_input_RSN)
    assert input_RSN.get("Seed") is None

    expected_input_subthreshold = {
        "Pattern": "SubThreshold",
        "PercentLess": 50.0
    }
    utils.check_is_subset(SimConfig.stimuli[2], expected_input_subthreshold)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "reports": {
                "soma_report": {
                    "type": "compartment",
                    "cells": "l4pc",
                    "variable_name": "v",
                    "sections": "soma",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 50.0
                },
                "compartment_report": {
                    "type": "compartment",
                    "cells": "l4pc",
                    "variable_name": "v",
                    "sections": "all",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "file_name": "my_compartment_report"
                }
            }
        }
    }
], indirect=True)
def test_parse_reports(create_tmp_simulation_config_file):
    SimConfig.init(create_tmp_simulation_config_file, {})
    expected_soma_report = {
        "Target": "l4pc",
        "Type": "compartment",
        "ReportOn": "v",
        "Compartments": "center",
        "Sections": "soma",
        "Scaling": "Area",
        "StartTime": 0.0,
        "EndTime": 50.0,
        "Dt": 0.1,
        "Enabled": True,
        "FileName": str(Path(SimConfig.run_conf["OutputRoot"]) / "soma_report.h5")
    }
    utils.check_is_subset(SimConfig.reports["soma_report"], expected_soma_report)

    expected_compartment_report = {
        "Target": "l4pc",
        "Type": "compartment",
        "ReportOn": "v",
        "Compartments": "all",
        "Sections": "all",
        "Scaling": "Area",
        "StartTime": 0.0,
        "EndTime": 10.0,
        "Dt": 0.1,
        "Enabled": True,
        "FileName": str(
            Path(SimConfig.run_conf["OutputRoot"]) / "my_compartment_report.h5"
        )
    }
    utils.check_is_subset(SimConfig.reports["compartment_report"], expected_compartment_report)
