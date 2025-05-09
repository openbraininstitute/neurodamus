import logging

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import RINGTEST_DIR

import neurodamus.core.stimuli as st
import neurodamus.stimulus_manager as smng
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
    np.isclose(stimulus.amp_start, 6.6)
    np.isclose(stimulus.amp_end, 6.6)
    assert len(stimulus.stimList) == 3
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(signal_source.stim_vec, [0, 6.6, 6.6, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 0, 10, 10])


def test_pulse(ringtest_stimulus_manager):
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
    np.isclose(stimulus.amp_start, 59.4)
    np.isclose(stimulus.amp_end, 59.4)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(signal_source.stim_vec, [0, 59.4, 59.4, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 0, 10, 10])


def test_hyperpolarizing(ringtest_stimulus_manager):
    stim_info = {
        "Pattern": "Hyperpolarizing",
        "Mode": "Current",
        "Duration": 10,
        "Delay": 0,
    }
    ringtest_stimulus_manager.interpret(target_onecell, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, smng.Hyperpolarizing)
    np.isclose(stimulus.amp_start, 88)
    np.isclose(stimulus.amp_end, 88)
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(signal_source.stim_vec, [0, 88, 88, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 0, 10, 10])


def test_seclamp(caplog, ringtest_stimulus_manager):
    stim_info = {
        "Pattern": "SEClamp",
        "Mode": "Voltage",
        "Voltage": 5.0,
        "SeriesResistance": 0.1,
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
    np.isclose(signal_source.rs, 0.1)
    np.isclose(signal_source.dur1, 10)
    np.isclose(signal_source.amp1, 5)


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
    np.isclose(stimulus.mean, 5)
    np.isclose(stimulus.var, 0.1)
    np.isclose(stimulus.dt, 0.5)  # no Dt in SONATA spec for noise, so always default 0.5
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
