"""Unit tests for StimulusManager to handline various stimulus types"""

import logging

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import RINGTEST_DIR

import neurodamus.core.stimuli as st
import neurodamus.stimulus_manager as smng
from neurodamus.core.configuration import ConfigurationError
from neurodamus.node import Node

target_name = "RingA"
target_onecell = "RingA_oneCell"


@pytest.fixture
def ringtest_stimulus_manager():
    n = Node(str(RINGTEST_DIR / "simulation_config.json"))
    n.load_targets()
    n.create_cells()
    # for testing certain stimuli, set dummy values to cell threshold_current and holding_current
    for cell in n.circuits.get_node_manager("RingA").cells:
        cell.setThreshold(66)
        cell.setHypAmp(88)
    return smng.StimulusManager(n._target_manager)


def test_linear(ringtest_stimulus_manager):
    """Linear Stimulus"""

    stim_info = {
        "Pattern": "Linear",
        "Mode": "Current",
        "AmpStart": 1.0,
        "AmpEnd": 10.0,
        "Duration": 20,
        "Delay": 5,
    }
    ringtest_stimulus_manager.interpret(target_name, stim_info)
    assert len(ringtest_stimulus_manager._stimulus) == 1
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.Linear)
    assert len(stimulus.stimList) == 3
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    assert not signal_source._represents_physical_electrode
    npt.assert_allclose(signal_source.stim_vec, [0, 0, 1, 10, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 5, 5, 25, 25])


def test_reletive_linear(ringtest_stimulus_manager):
    """RelativeLinear Stimulus"""

    target_name = "RingA"
    stim_info = {
        "Pattern": "RelativeLinear",
        "Mode": "Current",
        "PercentStart": 10,
        "PercentEnd": 10.0,
        "Duration": 10,
        "Delay": 0,
    }
    ringtest_stimulus_manager.interpret(target_name, stim_info)
    assert len(ringtest_stimulus_manager._stimulus) == 1
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.RelativeLinear)
    assert np.isclose(stimulus.amp_start, 6.6)
    assert np.isclose(stimulus.amp_end, 6.6)
    assert len(stimulus.stimList) == 3
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(signal_source.stim_vec, [0, 6.6, 6.6, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 0, 10, 10])


def test_pulse(ringtest_stimulus_manager):
    """Pulse Stimulus"""

    stim_info = {
        "Pattern": "Pulse",
        "Mode": "Current",
        "AmpStart": 10,
        "Frequency": 50.0,
        "Width": 1,
        "Duration": 50,
        "Delay": 5,
        "RepresentsPhysicalElectrode": True,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    assert len(ringtest_stimulus_manager._stimulus) == 1
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.Pulse)
    assert len(stimulus.stimList) == 1
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    assert signal_source._represents_physical_electrode
    npt.assert_allclose(
        signal_source.stim_vec,
        [0, 0, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10, 0, 0],
    )
    npt.assert_allclose(
        signal_source.time_vec,
        [0, 5, 5, 6, 6, 25, 25, 26, 26, 45, 45, 46, 46, 55],
    )


def test_sinusoidal(ringtest_stimulus_manager):
    """Sinusoidal Stimulus"""

    stim_info = {
        "Pattern": "Sinusoidal",
        "Mode": "Current",
        "AmpStart": 20,
        "Frequency": 1 / (2 * np.pi),
        "Duration": 10,
        "Delay": 0,
        "Dt": 1,
        "RepresentsPhysicalElectrode": True,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.Sinusoidal)
    assert len(stimulus.stimList) == 1
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec,
        [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0],
        atol=1e-5,
    )
    npt.assert_allclose(signal_source.time_vec, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])


def test_subthreshold(ringtest_stimulus_manager):
    """SubThreshold Stimulus"""

    stim_info = {
        "Pattern": "SubThreshold",
        "Mode": "Current",
        "PercentLess": 10,
        "Duration": 10,
        "Delay": 0,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.SubThreshold)
    assert np.isclose(stimulus.amp_start, 59.4)
    assert np.isclose(stimulus.amp_end, 59.4)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(signal_source.stim_vec, [0, 59.4, 59.4, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 0, 10, 10])


def test_hyperpolarizing(ringtest_stimulus_manager):
    """Hyperpolarizing Stimulus"""

    stim_info = {
        "Pattern": "Hyperpolarizing",
        "Mode": "Current",
        "Duration": 10,
        "Delay": 0,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.Hyperpolarizing)
    assert np.isclose(stimulus.amp_start, 88)
    assert np.isclose(stimulus.amp_end, 88)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(signal_source.stim_vec, [0, 88, 88, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 0, 10, 10])


def test_seclamp(caplog, ringtest_stimulus_manager):
    """SECLamp Stimlus"""

    stim_info = {
        "Pattern": "SEClamp",
        "Mode": "Voltage",
        "Voltage": 5.0,
        "RS": 0.1,
        "Duration": 10,
        "Delay": 1,
    }

    with caplog.at_level(logging.WARNING):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)
        assert "SEClamp ignores delay" in caplog.text

    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.SEClamp)
    signal_source = stimulus.stimList[0]
    assert signal_source.hname() == "SEClamp[0]"
    assert np.isclose(signal_source.rs, 0.1)
    assert np.isclose(signal_source.dur1, 10)
    assert np.isclose(signal_source.amp1, 5)


def test_noise(ringtest_stimulus_manager):
    """Test two Noise stimulus settings, one is Mean, the other is MeanPercent"""
    stim_info1 = {
        "Pattern": "Noise",
        "Mode": "Current",
        "Mean": 5.0,
        "Variance": 0.1,
        "Duration": 5,
        "Delay": 1,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info1)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.Noise)
    assert np.isclose(stimulus.mean, 5)
    assert np.isclose(stimulus.var, 0.1)
    assert np.isclose(stimulus.dt, 0.5)  # no Dt in SONATA spec for noise, so always default 0.5
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec,
        [
            0,
            0,
            4.881986,
            5.089319,
            4.790715,
            5.095573,
            5.490706,
            5.18951,
            5.373385,
            5.192794,
            5.219076,
            5.072542,
            5.301134,
            0,
        ],
    )
    npt.assert_allclose(
        signal_source.time_vec, [0, 1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6]
    )

    stim_info2 = {
        "Pattern": "Noise",
        "Mode": "Current",
        "MeanPercent": 10,
        "Variance": 0.1,
        "Duration": 5,
        "Delay": 1,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info2)
    stimulus = ringtest_stimulus_manager._stimulus[1]
    assert isinstance(stimulus, smng.Noise)
    assert np.isclose(stimulus.mean, 6.6)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec,
        [
            0,
            0,
            6.581311,
            6.899133,
            7.000418,
            6.623692,
            6.859104,
            6.43775,
            6.369685,
            6.825659,
            6.487686,
            6.438033,
            6.957322,
            0,
        ],
    )
    npt.assert_allclose(
        signal_source.time_vec, [0, 1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6]
    )


def test_shot_noise(ringtest_stimulus_manager):
    """Test ShotNoise Current and Conductance Modes"""

    # Current Mode
    stim_info = {
        "Pattern": "ShotNoise",
        "Mode": "Current",
        "Duration": 4,
        "Delay": 1,
        "Dt": 0.5,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "AmpMean": 20,
        "AmpVar": 10,
        "Rate": 1000,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.ShotNoise)
    assert np.isclose(stimulus.amp_mean, 20)
    assert np.isclose(stimulus.amp_var, 10)
    assert np.isclose(stimulus.rate, 1000)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec,
        [0, 0, 0, 0, 0, 0, 15.809228, 5.922412, 2.179451, 0.80178, 16.902339, 0],
        atol=1e-6,
    )
    npt.assert_allclose(signal_source.time_vec, [0, 1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5])

    # Conductance Mode
    stim_info["Mode"] = "Conductance"
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[1]
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.ConductanceSource)
    npt.assert_allclose(
        signal_source.stim_vec,
        [
            0,
            0,
            0,
            0,
            10.789501,
            4.041935,
            18.436711,
            6.896693,
            15.431213,
            5.763706,
            2.120934,
            0,
        ],
        atol=1e-6,
    )
    npt.assert_allclose(signal_source.time_vec, [0, 1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5])


def test_relative_shot_noise(ringtest_stimulus_manager):
    """Test RelativeShotNoise Current and Conductance Modes"""

    # Current Mode
    stim_info = {
        "Pattern": "RelativeShotNoise",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "MeanPercent": 20,
        "SDPercent": 0.5,
        "RelativeSkew": 0.1,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.RelativeShotNoise)
    assert np.isclose(stimulus.amp_mean, 0.011917)
    assert np.isclose(stimulus.amp_var, 1.57793e-5)
    assert np.isclose(stimulus.rate, 1481481.48)
    assert np.isclose(stimulus.rel_skew, 0.1)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec,
        [
            0,
            0,
            0,
            2.244623,
            5.828031,
            8.581348,
            9.838848,
            10.419317,
            11.512229,
            11.929373,
            0,
        ],
        atol=1e-6,
    )
    npt.assert_allclose(signal_source.time_vec, [0, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2])

    # Conductance Mode, cell must contain attribute input_resistance
    stim_info["Mode"] = "Conductance"
    with pytest.raises(AttributeError, match="input_resistance"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    # add dummy "input_resistance" in the target cell, try again
    cell_manager = ringtest_stimulus_manager._target_manager._cell_manager
    cell = cell_manager.get_cell(2)
    cell.extra_attrs["input_resistance"] = 0.01
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[1]
    assert np.isclose(stimulus.amp_mean, 0.01805599)
    assert np.isclose(stimulus.amp_var, 3.62243e-05)
    assert np.isclose(stimulus.rate, 1481481.48)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.ConductanceSource)
    npt.assert_allclose(
        signal_source.stim_vec,
        [
            0,
            0,
            0,
            3.020382,
            8.752334,
            12.78625,
            15.184817,
            16.831108,
            16.9444,
            17.017716,
            0,
        ],
        atol=1e-6,
    )
    npt.assert_allclose(signal_source.time_vec, [0, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2])


def test_absolute_shot_noise(ringtest_stimulus_manager):
    """Test AbsoluteShotNoise Current and Conductance Modes"""

    # Current Mode
    stim_info = {
        "Pattern": "AbsoluteShotNoise",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "Mean": 10,
        "Sigma": 1,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.AbsoluteShotNoise)
    assert np.isclose(stimulus.amp_mean, 0.08024884)
    assert np.isclose(stimulus.amp_var, 0.0064399)
    assert np.isclose(stimulus.rate, 166666.67)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec, [0, 0, 0, 2.987401, 5.837131, 7.698501, 0], atol=1e-6
    )
    npt.assert_allclose(signal_source.time_vec, [0, 0, 0.5, 1.0, 1.5, 2, 2])

    # Conductance Mode
    stim_info["Mode"] = "Conductance"
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[1]
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.ConductanceSource)
    npt.assert_allclose(
        signal_source.stim_vec, [0, 0, 0, 1.961163, 4.489483, 7.321958, 0], atol=1e-6
    )
    npt.assert_allclose(signal_source.time_vec, [0, 0, 0.5, 1.0, 1.5, 2, 2])


def test_ornstein_uhlenbeck(ringtest_stimulus_manager):
    """Test OrnsteinUhlenbeck Current and Conductance Modes"""

    # Current Mode
    stim_info = {
        "Pattern": "OrnsteinUhlenbeck",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "Tau": 2.7,
        "Reversal": 0.1,
        "Mean": 10,
        "Sigma": 1,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.OrnsteinUhlenbeck)
    assert np.isclose(stimulus.tau, 2.7)
    assert stimulus.seed is None
    assert np.isclose(stimulus.mean, 10)
    assert np.isclose(stimulus.sigma, 1)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec, [0, 10, 10.847061, 11.000316, 10.01365, 9.830591, 0], atol=1e-6
    )
    npt.assert_allclose(signal_source.time_vec, [0, 0, 0.5, 1.0, 1.5, 2, 2])

    # Conductance Mode
    stim_info["Mode"] = "Conductance"
    stim_info["Seed"] = 1234
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[1]
    assert np.isclose(stimulus.tau, 2.7)
    assert np.isclose(stimulus.seed, 1234)
    assert np.isclose(stimulus.mean, 10)
    assert np.isclose(stimulus.sigma, 1)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.ConductanceSource)
    npt.assert_allclose(
        signal_source.stim_vec, [0.1, 10, 9.637036, 9.793568, 10.338998, 10.374755, 0.1], atol=1e-6
    )
    npt.assert_allclose(signal_source.time_vec, [0, 0, 0.5, 1.0, 1.5, 2, 2])


def test_relative_ornstein_uhlenbeck(ringtest_stimulus_manager):
    """Test RelativeOrnsteinUhlenbeck Current and Conductance Modes"""

    # Current Mode
    stim_info = {
        "Pattern": "RelativeOrnsteinUhlenbeck",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "Tau": 2.7,
        "Reversal": 0.1,
        "MeanPercent": 10,
        "SDPercent": 1,
        "Seed": 5678,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.RelativeOrnsteinUhlenbeck)
    assert np.isclose(stimulus.tau, 2.7)
    assert np.isclose(stimulus.seed, 5678)
    assert np.isclose(stimulus.mean, 6.6)
    assert np.isclose(stimulus.sigma, 0.66)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec, [0.0, 6.6, 6.168779, 6.035724, 6.643255, 6.56104, 0.0]
    )
    npt.assert_allclose(signal_source.time_vec, [0, 0, 0.5, 1.0, 1.5, 2, 2])

    # Conductance Mode, cell must contain attribute input_resistance
    stim_info["Mode"] = "Conductance"
    with pytest.raises(AttributeError, match="input_resistance"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    # add dummy "input_resistance" in the target cell, try again
    cell_manager = ringtest_stimulus_manager._target_manager._cell_manager
    cell = cell_manager.get_cell(2)
    cell.extra_attrs["input_resistance"] = 0.02
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[1]
    assert np.isclose(stimulus.tau, 2.7)
    assert np.isclose(stimulus.mean, 5)
    assert np.isclose(stimulus.sigma, 0.5)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.ConductanceSource)
    npt.assert_allclose(
        signal_source.stim_vec, [0.1, 5, 4.839424, 5.020966, 5.083812, 4.941851, 0.1]
    )
    npt.assert_allclose(signal_source.time_vec, [0, 0, 0.5, 1.0, 1.5, 2, 2])


def test_error_unknown_pattern(ringtest_stimulus_manager):
    stim_info = {
        "Pattern": "Unknown",
    }
    with pytest.raises(ConfigurationError, match="No implementation for Stimulus Unknow"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)


def test_errorhandling_ornsteinuhlenbeck(ringtest_stimulus_manager):
    """Test various checks on OrnsteinUhlenbeck parameters"""
    stim_info = {
        "Pattern": "OrnsteinUhlenbeck",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Tau": 2.7,
        "Reversal": 0.1,
        "Mean": 10,
        "Sigma": 1,
        "Dt": -1,
    }
    with pytest.raises(Exception, match="OrnsteinUhlenbeck time-step must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "OrnsteinUhlenbeck",
        "Mode": "Voltage",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "Tau": 2.7,
        "Reversal": 0.1,
        "Mean": 10,
        "Sigma": 1,
    }
    with pytest.raises(
        Exception, match="OrnsteinUhlenbeck must be used with mode Current or Conductance"
    ):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "OrnsteinUhlenbeck",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "Tau": -2.7,
        "Reversal": 0.1,
        "Mean": 10,
        "Sigma": 1,
    }
    with pytest.raises(Exception, match="OrnsteinUhlenbeck relaxation time must be non-negative"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "OrnsteinUhlenbeck",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "Tau": 2.7,
        "Reversal": 0.1,
        "Mean": 10,
        "Sigma": -1,
    }
    with pytest.raises(Exception, match="OrnsteinUhlenbeck standard deviation must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)


def test_errorhandling_relative_ornsteinuhlenbeck(ringtest_stimulus_manager):
    """Test various checks on RelativeOrnsteinUhlenbeck parameters"""

    stim_info = {
        "Pattern": "RelativeOrnsteinUhlenbeck",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "Tau": 2.7,
        "Reversal": 0.1,
        "MeanPercent": 10,
        "SDPercent": -1,
        "Seed": 5678,
    }
    with pytest.raises(
        Exception, match="RelativeOrnsteinUhlenbeck standard deviation must be positive"
    ):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)


def test_errorhandling_sinusoidal(ringtest_stimulus_manager):
    """Test various checks on Sinusoidal parameters"""

    stim_info = {
        "Pattern": "Sinusoidal",
        "Mode": "Current",
        "AmpStart": 20,
        "Frequency": 1 / (2 * np.pi),
        "Duration": 10,
        "Delay": 0,
        "Dt": -1,
    }
    with pytest.raises(Exception, match="Sinusoidal time-step must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)


def test_errorhandling_noise(ringtest_stimulus_manager):
    """Test various checks on Noise parameters"""

    stim_info = {
        "Pattern": "Noise",
        "Mode": "Current",
        "Mean": 5.0,
        "Variance": 0.1,
        "Duration": 5,
        "Delay": 1,
        "Dt": -1,
    }
    with pytest.raises(Exception, match="Noise time-step must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "Noise",
        "Mode": "Current",
        "Mean": 5.0,
        "Variance": -0.1,
        "Duration": 5,
        "Delay": 1,
        "Dt": 1,
    }
    with pytest.raises(Exception, match="Noise variance must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "Noise",
        "Mode": "Current",
        "MeanPercent": 5.0,
        "Variance": -0.1,
        "Duration": 5,
        "Delay": 1,
        "Dt": 1,
    }
    with pytest.raises(Exception, match="Noise variance percent must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)


def test_errorhandling_shotnoise(ringtest_stimulus_manager):
    """Test various checks on ShotNoise parameters"""

    stim_info = {
        "Pattern": "ShotNoise",
        "Mode": "Current",
        "Duration": 4,
        "Delay": 1,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "AmpMean": 20,
        "AmpVar": 10,
        "Rate": 1000,
        "Dt": -1,
    }
    with pytest.raises(Exception, match="ShotNoise time-step must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "ShotNoise",
        "Mode": "Voltage",
        "Duration": 4,
        "Delay": 1,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "AmpMean": 20,
        "AmpVar": 10,
        "Rate": 1000,
        "Dt": -1,
    }
    with pytest.raises(Exception, match="ShotNoise must be used with mode Current or Conductance"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "ShotNoise",
        "Mode": "Current",
        "Duration": 4,
        "Delay": 1,
        "RiseTime": 0.5,
        "DecayTime": 0.1,
        "AmpMean": 20,
        "AmpVar": 10,
        "Rate": 1000,
    }
    with pytest.raises(
        Exception, match="ShotNoise bi-exponential rise time must be smaller than decay time"
    ):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "ShotNoise",
        "Mode": "Current",
        "Duration": 4,
        "Delay": 1,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "AmpMean": 0,
        "AmpVar": 10,
        "Rate": 1000,
    }
    with pytest.raises(Exception, match="ShotNoise amplitude mean must be non-zero"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "ShotNoise",
        "Mode": "Current",
        "Duration": 4,
        "Delay": 1,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "AmpMean": 20,
        "AmpVar": -10,
        "Rate": 1000,
    }
    with pytest.raises(Exception, match="ShotNoise amplitude variance must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)


def test_errorhandling_relative_shotnoise(ringtest_stimulus_manager):
    """Test various checks on RelativeShotNoise parameters"""

    stim_info = {
        "Pattern": "RelativeShotNoise",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "MeanPercent": 20,
        "SDPercent": -0.5,
    }
    with pytest.raises(Exception, match="RelativeShotNoise stdev percent must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "RelativeShotNoise",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "MeanPercent": 20,
        "SDPercent": 0.5,
        "RelativeSkew": 1.5,
    }
    with pytest.raises(Exception, match=r"RelativeShotNoise relative skewness must be in \[0,1\]"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)


def test_errorhandling_absolute_shot_noise(ringtest_stimulus_manager):
    """Test various checks on AbsoluteShotNoise parameters"""

    stim_info = {
        "Pattern": "AbsoluteShotNoise",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "Mean": 10,
        "Sigma": -1,
    }
    with pytest.raises(Exception, match="AbsoluteShotNoise stdev must be positive"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)

    stim_info = {
        "Pattern": "AbsoluteShotNoise",
        "Mode": "Current",
        "Duration": 2,
        "Delay": 0,
        "Dt": 0.5,
        "RiseTime": 0.1,
        "DecayTime": 0.5,
        "Mean": 10,
        "Sigma": 1,
        "RelativeSkew": 1.5,
    }
    with pytest.raises(Exception, match=r"AbsoluteShotNoise relative skewness must be in \[0,1\]"):
        ringtest_stimulus_manager.interpret(target_onecell, stim_info)
