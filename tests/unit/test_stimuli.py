"""A collection of tests for advanced stimulus generated with the help of Neuron."""

import pytest
import neurodamus.core.stimuli as st
from neurodamus.core.random import Random123
import numpy as np


class TestSignalSource:
    def setup_method(self):
        rng = Random123(1, 2, 3)
        self.base_delay = 1.0
        self.base_amp = 2.0
        self.stim = st.SignalSource(rng=rng, base_amp=self.base_amp, delay=self.base_delay)

    def test_reset(self):
        """Reset from delay.

        Test that something changed
        """
        assert list(self.stim.time_vec) == [0]
        assert list(self.stim.stim_vec) == [self.base_amp]
        self.stim.reset()
        assert list(self.stim.time_vec) == []
        assert list(self.stim.stim_vec) == []

    def test_delay(self):
        """Add delay.

        Verify that time_vec did not change
        """
        add_delay = 2.0
        l_time_vec = len(self.stim.time_vec)
        self.stim.delay(add_delay)
        assert len(self.stim.time_vec) == l_time_vec
        assert len(self.stim.stim_vec) == l_time_vec
        assert self.stim._cur_t == pytest.approx(self.base_delay + add_delay)

    def test_add_segment(self):
        """Add 2 segments."""
        A1, A2, A3 = 1.0, 2.0, 4.0
        D1, D2 = 2.0, 7.0
        self.stim.add_segment(A1, D1, A2)
        self.stim.add_segment(A3, D2)
        assert np.allclose(self.stim.stim_vec, [self.base_amp, A1, A2, A3, A3])
        expected = np.array([-self.base_delay, 0.0, D1, D1, D1 + D2]) + self.base_delay
        assert np.allclose(self.stim.time_vec, expected)

    def test_add_pulse_and_ramp(self):
        """Add a pulse/ramp segment and verify correct amplitude and timing."""
        A1, A2, A3, new_base_amp = 3.0, 4.0, 2.5, 5.0
        D1, D2 = 0.5, 0.8
        self.stim.add_pulse(A1, D1)
        self.stim.add_ramp(A2, A3, D2, base_amp=new_base_amp)

        expected = (
            np.array([-self.base_delay, 0.0, 0.0, D1, D1, D1, D1, D1 + D2, D1 + D2])
            + self.base_delay
        )
        assert np.allclose(self.stim.time_vec, expected)
        expected = [
            self.base_amp,
            self.base_amp,
            A1,
            A1,
            self.base_amp,
            new_base_amp,
            A2,
            A3,
            new_base_amp,
        ]
        assert np.allclose(list(self.stim.stim_vec), expected)

    @pytest.mark.parametrize("base_amp", [-1, 0, 1.5])
    def test_pulse_diff_base(self, base_amp):
        """Sweep test of `add_pulse` with varying `base_amp` values, verifying expected time and
        stimulus vectors."""
        self.stim.add_pulse(1.2, 10, base_amp=base_amp)
        expected = np.array([-self.base_delay, 0, 0, 10, 10]) + self.base_delay
        assert np.allclose(list(self.stim.time_vec), expected)
        assert np.allclose(list(self.stim.stim_vec), [self.base_amp, base_amp, 1.2, 1.2, base_amp])

    def test_add_train(self):
        """Add train of pulses.

        Check for negative delays.
        """
        amp, frequency, pulse_duration, total_duration, base_amp = (
            0.3,
            5000,
            3,
            4.5,
            0.22,
        )

        with pytest.raises(ValueError):
            self.stim.add_train(
                amp=amp,
                frequency=frequency,
                pulse_duration=pulse_duration,
                total_duration=total_duration,
                base_amp=base_amp,
            )

        frequency, pulse_duration = 500, 0.6
        self.stim.add_train(
            amp=amp,
            frequency=frequency,
            pulse_duration=pulse_duration,
            total_duration=total_duration,
            base_amp=base_amp,
        )
        assert np.allclose(
            self.stim.time_vec,
            [0.0, 1.0, 1.0, 1.6, 1.6, 3.0, 3.0, 3.6, 3.6, 5.0, 5.0, 5.5, 5.5, 5.5],
        )
        assert np.allclose(
            self.stim.stim_vec,
            [
                2.0,
                0.22,
                0.3,
                0.3,
                0.22,
                0.22,
                0.3,
                0.3,
                0.22,
                0.22,
                0.3,
                0.3,
                0.22,
                0.22,
            ],
        )

    def test_long_add_train(self):
        """Test `add_train` with long duration, verifying correct time and stimulus vectors."""
        self.stim.add_train(1.2, 10, 20, 350)
        # At 10Hz pulses have T=100ms
        # We end up with 4 pulses, the last one with reduced rest phase
        expected = (
            np.array(
                [
                    -self.base_delay,
                    0,
                    0,
                    20,
                    20,
                    100,
                    100,
                    120,
                    120,
                    200,
                    200,
                    220,
                    220,
                    300,
                    300,
                    320,
                    320,
                    350,
                ]
            )
            + self.base_delay
        )
        assert np.allclose(self.stim.time_vec, expected)
        assert np.allclose(
            self.stim.stim_vec,
            [self.base_amp] + [self.base_amp, 1.2, 1.2, self.base_amp] * 4 + [self.base_amp],
        )

    def test_add_sin(self):
        """Test `add_sin` with short duration, ensuring expected time and sinusoidal stimulus
        vectors."""
        self.stim.add_sin(
            1,
            0.1,
            10000,
        )
        expected = np.array([-self.base_delay, 0, 0.025, 0.05, 0.075, 0.1, 0.1]) + self.base_delay
        assert np.allclose(self.stim.time_vec, expected)
        assert np.allclose(self.stim.stim_vec, [self.base_amp, 0, 1, 0, -1, 0, self.base_amp])

    def test_long_add_sin(self):
        """Test `add_sin` with longer duration, validating time and sinusoidal stimulus vectors."""
        self.stim.add_sin(1, 200, 10, 25)
        expected = (
            np.array([-self.base_delay, 0, 25, 50, 75, 100, 125, 150, 175, 200, 200])
            + self.base_delay
        )
        assert np.allclose(self.stim.time_vec, expected)
        assert np.allclose(
            self.stim.stim_vec, [self.base_amp] + [0, 1, 0, -1] * 2 + [0, self.base_amp]
        )

    def test_add_pulses(self):
        """Test `add_pulses` with multiple amplitudes, verifying time and stimulus vectors."""
        self.stim.add_pulses(0.5, 1, 2, 3, 4, base_amp=0.1)
        expected = (
            np.array([-self.base_delay, 0, 0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2]) + self.base_delay
        )
        assert np.allclose(self.stim.time_vec, expected)
        assert np.allclose(self.stim.stim_vec, [self.base_amp, 0.1, 1, 1, 2, 2, 3, 3, 4, 4, 0.1])

    def test_add_noise(self):
        """Test `add_noise` with given duration and amplitude, checking time and stimulus
        vectors."""
        self.stim.add_noise(0.5, 0.1, 5)
        expected = (
            np.array([-self.base_delay, 0, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5])
            + self.base_delay
        )
        assert np.allclose(self.stim.time_vec, expected)
        assert np.allclose(
            self.stim.stim_vec,
            [
                self.base_amp,
                self.base_amp,
                0.70681968,
                0.56316322,
                0.5539058,
                0.6810689,
                0.20896532,
                1.00691217,
                0.78783759,
                0.68817496,
                -6.4286609e-05,
                0.21165959,
                0.03874813,
                self.base_amp,
            ],
        )

    def test_add_shot_noise(self):
        """Test Poisson shot noise signal with gamma-distributed amplitudes and bi-exponential
        impulse response."""
        self.stim.add_shot_noise(4.0, 0.4, 2e3, 40e-3, 16e-4, 2)
        base_time_vec = np.array([0, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0])
        expected_time_vec = base_time_vec + self.base_delay
        expected_time_vec = np.concatenate(([0], expected_time_vec))
        assert np.allclose(self.stim.time_vec, expected_time_vec)
        expected_stim_vec = [
            self.base_amp,
            self.base_amp,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0700357,
            0.1032799,
            0.1170881,
            0.1207344,
            self.base_amp,
        ]
        assert np.allclose(self.stim.stim_vec, expected_stim_vec)

    def test_add_shot_noise_no_rng(self):
        """Try the same as add_snot_noise but with the standard rng."""
        no_rng_stim = st.SignalSource(base_amp=self.base_amp, delay=self.base_delay)
        no_rng_stim.add_shot_noise(4.0, 0.4, 2e3, 40e-3, 16e-4, 2)
        expected = (
            np.array(
                [
                    -self.base_delay,
                    0,
                    0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    1.25,
                    1.5,
                    1.75,
                    2.0,
                    2.0,
                ]
            )
            + self.base_delay
        )
        assert np.allclose(no_rng_stim.time_vec, expected)
        expected = [
            self.base_amp,
            self.base_amp,
            0.0,
            0.0,
            0.06718156,
            0.09907093,
            0.11231636,
            0.11581408,
            0.11431186,
            0.1103378,
            0.10523273,
            self.base_amp,
        ]
        assert np.allclose(no_rng_stim.stim_vec, expected)

    def test_add_shot_noise_negative_A(self):
        """Test with negative amplitude."""
        self.stim.add_shot_noise(4.0, 0.4, 2e3, -40e-3, 16e-4, 2)
        base_time_vec = np.array([0, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0])
        expected_time_vec = base_time_vec + self.base_delay
        expected_time_vec = np.concatenate(([0], expected_time_vec))
        assert np.allclose(self.stim.time_vec, expected_time_vec)
        expected_stim_vec = [
            self.base_amp,
            self.base_amp,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.0700357,
            -0.1032799,
            -0.1170881,
            -0.1207344,
            self.base_amp,
        ]
        assert np.allclose(self.stim.stim_vec, expected_stim_vec)

    def test_add_shot_noise_wrong_inputs(self):
        """The edge case where a shot noise has same tau rise and decay is not implemented yet."""
        with pytest.raises(NotImplementedError):
            st.SignalSource.shot_noise(1.0, 1.0, 1.0, 1.0, 1.0, 100, dt=0.25, base_amp=0.0)

    def test_ornstein_uhlenbeck(self):
        """Test the OU process."""
        self.stim.add_ornstein_uhlenbeck(2.8, 0.0042, 0.029, 2)
        base_time_vec = np.array([0, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0])
        expected_time_vec = base_time_vec + self.base_delay
        expected_time_vec = np.concatenate(([0], expected_time_vec))
        assert np.allclose(self.stim.time_vec, expected_time_vec)
        expected_stim_vec = [
            self.base_amp,
            self.base_amp,
            0.029,
            0.02933925,
            0.02959980,
            0.03052109,
            0.02882802,
            0.03156533,
            0.03289219,
            0.03357043,
            0.03049419,
            self.base_amp,
        ]
        assert np.allclose(self.stim.stim_vec, expected_stim_vec)

    def test_ornstein_uhlenbeck_white_noise(self):
        """Test OU process when tau is too small and we add simple white noise."""
        self.stim.add_ornstein_uhlenbeck(0.5e-9, 0.0042, 0.029, 2)
        base_time_vec = np.array([0, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0])
        expected_time_vec = base_time_vec + self.base_delay
        expected_time_vec = np.concatenate(([0], expected_time_vec))
        assert np.allclose(self.stim.time_vec, expected_time_vec)
        expected_stim_vec = [
            self.base_amp,
            self.base_amp,
            0.031746889227565855,
            0.029838906502493365,
            0.029715953168352027,
            0.0314048785566717,
            0.0251346035896446,
            0.03573258758803672,
            0.03282293402995273,
            0.031499258125166095,
            0.022358363086789193,
            self.base_amp,
        ]
        assert np.allclose(
            self.stim.stim_vec, expected_stim_vec
        ), f"{list(self.stim.stim_vec)}, {list(expected_stim_vec)}"

    def test_plot(self):
        import matplotlib

        matplotlib.use("agg")
        self.stim.plot([10])

    def test_stacked(self):
        """Assert each individual stim, namely noise, starts and returns to zero."""
        stim = self.stim
        stim.add_noise(0.5, 0.1, 2)  # dt is 0.5
        STIM1_SAMPLES = 8  # 1+4+1+2
        assert stim.time_vec.size() == stim.stim_vec.size() == STIM1_SAMPLES
        assert stim.time_vec[0] == 0.0
        assert stim.stim_vec[0] == self.base_amp
        assert stim.time_vec[-1] == self.base_delay + 2.0
        assert stim.stim_vec[-1] == self.base_amp

        stim.delay(2)
        stim.add_shot_noise(4.0, 0.4, 2e3, 0.5, 0.1, 2)  # dt is 0.025
        STIM2_SAMPLES = 11  # 8+1+2
        assert stim.time_vec.size() == stim.stim_vec.size() == STIM1_SAMPLES + STIM2_SAMPLES
        assert stim.time_vec[STIM1_SAMPLES + 1] == 4.0 + self.base_delay
        assert stim.stim_vec[STIM1_SAMPLES + 1] == 0.0
        assert stim.time_vec[-1] == 6.0 + self.base_delay
        assert stim.stim_vec[-1] == self.base_amp

    def test_class_methods(self):
        """Test class methods."""
        assert isinstance(st.SignalSource.pulse(5.0, 10, base_amp=0.0), st.SignalSource)
        assert isinstance(st.SignalSource.ramp(1.0, 5.0, 10, base_amp=0.0), st.SignalSource)
        assert isinstance(st.SignalSource.train(1.0, 50, 10, 100, base_amp=0.0), st.SignalSource)
        assert isinstance(st.SignalSource.sin(1.0, 100, 50, base_amp=0.0), st.SignalSource)
        assert isinstance(
            st.SignalSource.noise(0.0, 1.0, 100, dt=0.5, base_amp=0.0), st.SignalSource
        )
        assert isinstance(
            st.SignalSource.shot_noise(1.0, 2.0, 1.0, 1.0, 1.0, 100, dt=0.25, base_amp=0.0),
            st.SignalSource,
        )
        assert isinstance(
            st.SignalSource.ornstein_uhlenbeck(1.0, 1.0, 0.0, 100, dt=0.25, base_amp=0.0),
            st.SignalSource,
        )

    def test_not_implemented_methods(self):
        """Test that not implemented methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.stim.add_sinspec(0, 10)
        with pytest.raises(NotImplementedError):
            self.stim + st.SignalSource()


def create_ball_and_stick():
    from neurodamus.core import Neuron

    sec1 = Neuron.h.Section(name="sec1")
    sec1.nseg = 5
    soma = Neuron.h.Section(name="soma")
    soma.nseg = 5
    sec1.connect(soma)
    return sec1, soma


def clamp_attach_detach(stim, clamp_type):
    """Attach and detach a Clamp."""
    sec1, soma = create_ball_and_stick()
    assert clamp_type not in soma.psection()["point_processes"]
    assert clamp_type not in sec1.psection()["point_processes"]
    assert len(stim._all_sources) == 1
    assert len(stim._clamps) == 0
    clamp02 = stim.attach_to(soma, position=0.28)
    assert clamp_type in soma.psection()["point_processes"]
    assert clamp_type not in sec1.psection()["point_processes"]
    assert len(stim._clamps) == 1
    # we placed it at 0.28. Neuron snaps it to the center of the location
    # compartment. We have 5 real comartments. One center is in 0.3
    assert np.allclose(clamp02.clamp.get_loc(), 0.3)
    clamp02.detach()
    assert clamp_type not in soma.psection()["point_processes"]
    assert clamp_type not in sec1.psection()["point_processes"]
    assert len(stim._all_sources) == 1
    assert len(stim._clamps) == 0


class TestIClampSource:
    def setup_method(self):
        self.rng = Random123(1, 2, 3)
        self.base_delay = 1.0
        self.base_amp = 2.0
        self.stim = st.CurrentSource(
            rng=self.rng,
            base_amp=self.base_amp,
            delay=self.base_delay,
            represents_physical_electrode=True,
        )
        self.stim.add_segment(3, 4)

    def test_clamp_attach_detach(self):
        clamp_attach_detach(self.stim, "IClamp")

    def test_constant_attach_detach(self):
        sec1, soma = create_ball_and_stick()
        duration = 3.0
        assert "IClamp" not in soma.psection()["point_processes"]
        assert "IClamp" not in sec1.psection()["point_processes"]

        stim = st.CurrentSource.Constant(
            amp=self.base_amp,
            duration=duration,
            delay=self.base_delay,
            represents_physical_electrode=True,
        )
        clamp07 = stim.attach_to(soma, position=0.72)
        assert "IClamp" in soma.psection()["point_processes"]
        assert "IClamp" not in sec1.psection()["point_processes"]
        assert len(stim._clamps) == 1
        assert np.allclose(clamp07.clamp.get_loc(), 0.7)
        clamp07.detach()
        assert "IClamp" not in soma.psection()["point_processes"]
        assert "IClamp" not in sec1.psection()["point_processes"]
        assert len(stim._clamps) == 0


class TestMembraneCurrentSource:
    def setup_method(self):
        self.rng = Random123(1, 2, 3)
        self.base_delay = 1.0
        self.base_amp = 2.0
        self.stim = st.CurrentSource(
            rng=self.rng,
            base_amp=self.base_amp,
            delay=self.base_delay,
            represents_physical_electrode=False,
        )
        self.stim.add_segment(3, 4)

    def test_clamp_attach_detach(self):
        clamp_attach_detach(self.stim, "MembraneCurrentSource")


class TestSEClampSource:
    def setup_method(self):
        self.rng = Random123(1, 2, 3)
        self.base_delay = 1.0
        self.base_amp = 2.0
        self.stim = st.ConductanceSource(
            reversal=0.5, rng=self.rng, delay=self.base_delay, represents_physical_electrode=True
        )
        self.stim.add_segment(3, 4)

    def test_clamp_attach_detach(self):
        clamp_attach_detach(self.stim, "SEClamp")


class TestConductanceSource:
    def setup_method(self):
        self.rng = Random123(1, 2, 3)
        self.base_delay = 1.0
        self.base_amp = 2.0
        self.stim = st.ConductanceSource(
            reversal=0.5, rng=self.rng, delay=self.base_delay, represents_physical_electrode=False
        )
        self.stim.add_segment(3, 4)

    def test_clamp_attach_detach(self):
        clamp_attach_detach(self.stim, "ConductanceSource")
