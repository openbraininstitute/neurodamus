import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import RINGTEST_DIR

import neurodamus.core.stimuli as st
from neurodamus.node import Node
from neurodamus.stimulus_manager import Linear, Pulse, RelativeLinear, Sinusoidal, StimulusManager
from neurodamus.target_manager import TargetSpec


@pytest.fixture
def ringtest_stimulus_manager():
    n = Node(str(RINGTEST_DIR / "simulation_config.json"))
    n.load_targets()
    n.create_cells()
    return StimulusManager(n._target_manager)


def test_linear(ringtest_stimulus_manager):
    target = TargetSpec("RingA")
    stim_info = {
        "Pattern": "Linear",
        "Mode": "Current",
        "AmpStart": 1.0,
        "AmpEnd": 10.0,
        "Duration": 20,
        "Delay": 5,
    }
    ringtest_stimulus_manager.interpret(target, stim_info)
    assert len(ringtest_stimulus_manager._stimulus) == 1
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, Linear)
    assert len(stimulus.stimList) == 3
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    assert not signal_source._represents_physical_electrode
    npt.assert_allclose(signal_source.stim_vec, [0, 0, 1, 10, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 5, 5, 25, 25])


def test_reletive_linear(ringtest_stimulus_manager):
    target = TargetSpec("RingA")
    stim_info = {
        "Pattern": "RelativeLinear",
        "Mode": "Current",
        "PercentStart": 10,
        "PercentEnd": 10.0,
        "Duration": 10,
        "Delay": 0,
    }
    ringtest_stimulus_manager.interpret(target, stim_info)
    assert len(ringtest_stimulus_manager._stimulus) == 1
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, RelativeLinear)
    assert len(stimulus.stimList) == 3
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(signal_source.stim_vec, [0, 0, 0, 0])
    npt.assert_allclose(signal_source.time_vec, [0, 0, 10, 10])


def test_pulse(ringtest_stimulus_manager):
    target = TargetSpec("RingA_oneCell")
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
    ringtest_stimulus_manager.interpret(target, stim_info)
    assert len(ringtest_stimulus_manager._stimulus) == 1
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, Pulse)
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
    target = TargetSpec("RingA_oneCell")
    stim_info = {
        "Pattern": "Sinusoidal",
        "Mode": "Current",
        "AmpStart": 20,
        "Frequency": 1 / (2 * np.pi),
        "Duration": 10,
        "Delay": 0,
        "Dt": 1,
    }
    ringtest_stimulus_manager.interpret(target, stim_info)
    stimulus = ringtest_stimulus_manager._stimulus[0]
    assert isinstance(stimulus, Sinusoidal)
    assert len(stimulus.stimList) == 1
    signal_source = stimulus.stimList[0]
    assert isinstance(signal_source, st.CurrentSource)
    npt.assert_allclose(
        signal_source.stim_vec,
        [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0],
        atol=1e-5,
    )
    npt.assert_allclose(signal_source.time_vec.as_numpy(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
