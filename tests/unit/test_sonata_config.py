import pytest
from pathlib import Path

USECASE3 = Path(__file__).parent.parent.absolute() / "simulations" / "usecase3"
SONATA_CONF_FILE = str(USECASE3 / "simulation_sonata.json")

pytestmark = pytest.mark.forked


def test_parse_base():
    from neurodamus.io.sonata_config import SonataConfig
    raw_conf = SonataConfig(SONATA_CONF_FILE)
    assert raw_conf.run["random_seed"] == 1122
    assert raw_conf.parsedRun["BaseSeed"] == 1122


def test_SimConfig_from_sonata():
    from neurodamus.core.configuration import SimConfig
    SimConfig.init(SONATA_CONF_FILE, {})
    # RNGSettings in hoc correctly initialized from Sonata
    assert SimConfig.rng_info.getGlobalSeed() == 1122

    # run section
    assert SimConfig.run_conf['CircuitTarget'] == 'Mosaic'
    assert SimConfig.run_conf['Simulator'] == 'NEURON'
    assert SimConfig.run_conf['Duration'] == 50.0
    assert SimConfig.run_conf['Dt'] == 0.1
    assert SimConfig.run_conf['Celsius'] == 35
    assert SimConfig.run_conf['V_Init'] == -75
    assert SimConfig.run_conf['SpikeLocation'] == 'soma'

    # output section
    assert SimConfig.run_conf['SpikesFile'] == 'spikes.h5'
    assert SimConfig.run_conf['SpikesSortOrder'] == 'by_time'

    # reports section
    soma_report = SimConfig.reports['soma_report']
    assert soma_report['Target'] == 'l4pc'
    assert soma_report['Type'] == 'compartment'
    assert soma_report['ReportOn'] == 'v'
    assert soma_report['Compartments'] == 'center'
    assert soma_report['Sections'] == 'soma'
    assert soma_report['Scaling'] == 'Area'
    assert soma_report['StartTime'] == 0.0
    assert soma_report['EndTime'] == 50.0
    assert soma_report['Dt'] == 0.1
    assert soma_report['Enabled']
    compartment_report = SimConfig.reports['compartment_report']
    assert soma_report['FileName'] == str(Path(SimConfig.run_conf["OutputRoot"]) / "soma_report.h5")
    assert compartment_report['Target'] == 'l4pc'
    assert compartment_report['Type'] == 'compartment'
    assert compartment_report['ReportOn'] == 'v'
    assert compartment_report['Compartments'] == 'all'
    assert compartment_report['Sections'] == 'all'
    assert compartment_report['Scaling'] == 'Area'
    assert compartment_report['StartTime'] == 0.0
    assert compartment_report['EndTime'] == 10.0
    assert compartment_report['Dt'] == 0.1
    assert compartment_report['Enabled']
    assert compartment_report['FileName'] == str(
        Path(SimConfig.run_conf["OutputRoot"]) / "my_compartment_report.h5"
        )

    # conditions section
    conditions = list(SimConfig._simulation_config.Conditions.values())[0]
    assert conditions['init_depleted_ProbAMPANMDA_EMS'] is False
    assert conditions['minis_single_vesicle_ProbAMPANMDA_EMS'] is True
    assert conditions['randomize_Gaba_risetime'] == 'False'


contents = {
    "network": "circuit_config.json",
    "run":
    {
        "random_seed": 12345,
        "dt": 0.05,
        "tstop": 1000,
        "stimulus_seed": 1122
    },
    "conditions": {
        "spike_location": "AIS",
        "modifications": [
            {
                "name": "applyTTX",
                "node_set": "single",
                "type": "TTX"
            },
            {
                "name": "no_SK_E2",
                "node_set": "single",
                "type": "ConfigureAllSections",
                "section_configure": "%%s.gSK_E2bar_SK_E2 = 0"
            }
        ]
    },
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
    "inputs": {
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
        "hypamp_mosaic": {
            "node_set": "200_L5_PCs",
            "input_type": "current_clamp",
            "module": "hyperpolarizing",
            "delay": 0.0,
            "duration": 10000.0
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


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_data": contents,
        "src_dir": USECASE3
    }
], indirect=True)
def test_parse_seeds(create_tmp_simulation_config_file):
    from neurodamus.core.configuration import SimConfig

    SimConfig.init(create_tmp_simulation_config_file, {})
    assert SimConfig.rng_info.getGlobalSeed() == 12345
    assert SimConfig.rng_info.getStimulusSeed() == 1122
    assert SimConfig.rng_info.getIonChannelSeed() == 0
    assert SimConfig.rng_info.getMinisSeed() == 0
    assert SimConfig.rng_info.getSynapseSeed() == 0
    assert SimConfig.run_conf["SpikeLocation"] == "AIS"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_data": contents,
        "src_dir": USECASE3
    }
], indirect=True)
def test_parse_modifications(create_tmp_simulation_config_file):
    from neurodamus.core.configuration import SimConfig

    SimConfig.init(create_tmp_simulation_config_file, {})
    TTX_mod = SimConfig.modifications["applyTTX"]
    assert TTX_mod["Type"] == "TTX"
    assert TTX_mod["Target"] == "single"
    ConfigureAllSections_mod = SimConfig.modifications["no_SK_E2"]
    ConfigureAllSections_mod["Type"] = "ConfigureAllSections"
    ConfigureAllSections_mod["Target"] = "single"
    ConfigureAllSections_mod["SectionConfigure"] = "%s.gSK_E2bar_SK_E2 = 0"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_data": contents,
        "src_dir": USECASE3
    }
], indirect=True)
def test_parse_connections(create_tmp_simulation_config_file):
    from neurodamus.core.configuration import SimConfig

    SimConfig.init(create_tmp_simulation_config_file, {})
    conn = SimConfig.connections["GABAB_erev"]
    assert conn["Source"] == "Inhibitory"
    assert conn["Destination"] == "Mosaic"
    assert conn["Weight"] == 1.0
    assert conn.get("SpontMins") is None
    assert conn["Delay"] == 0
    assert conn["SynDelayOverride"] == 0.5
    assert conn.get("Modoverride") is None
    assert conn["SynapseConfigure"] == "%s.e_GABAA = -82.0 tau_d_GABAB_ProbGABAAB_EMS = 77"
    assert conn["NeuromodDtc"] == 100
    assert conn["NeuromodStrength"] == 0.75


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_data": contents,
        "src_dir": USECASE3
    }
], indirect=True)
def test_parse_inputs(create_tmp_simulation_config_file):
    from neurodamus.core.configuration import SimConfig

    SimConfig.init(create_tmp_simulation_config_file, {})
    input_hp = SimConfig.stimuli["hypamp_mosaic"]
    assert input_hp["Pattern"] == "Hyperpolarizing"
    assert input_hp["Mode"] == "Current"
    assert input_hp["Target"] == "200_L5_PCs"
    assert input_hp["Delay"] == 0.
    assert input_hp["Duration"] == 10000.0
    input_RSN = SimConfig.stimuli["RelativeShotNoise_L5E_inject"]
    assert input_RSN["Pattern"] == "RelativeShotNoise"
    assert input_RSN["Mode"] == "Current"
    assert input_RSN["DecayTime"] == 4.
    assert input_RSN["RiseTime"] == 0.4
    assert input_RSN["Delay"] == 0.
    assert input_RSN["Duration"] == 1000.
    assert input_RSN["MeanPercent"] == 70.
    assert input_RSN["RelativeSkew"] == 0.63
    assert input_RSN["SDPercent"] == 40.
    assert input_RSN["Dt"] == 0.25
    assert input_RSN.get("Seed") is None
    input_subthreshold = SimConfig.stimuli["subthreshould_mosaic"]
    assert input_subthreshold["Pattern"] == "SubThreshold"
    assert input_subthreshold["PercentLess"] == 50.0
