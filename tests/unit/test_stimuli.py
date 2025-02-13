"""
A collection of tests for advanced stimulus generated with the help of Neuron
"""

import pytest
import neurodamus.core.stimuli as st
from neurodamus.core.random import Random123


class TestSignalSource:
    def setup_method(self):
        rng = Random123(1, 2, 3)
        self.base_delay = 1.0
        self.base_amp = 2.0
        self.stim = st.SignalSource(
            rng=rng, base_amp=self.base_amp, delay=self.base_delay
        )

    def test_reset(self):
        """Reset from delay. Test that something changed"""
        assert list(self.stim.time_vec) == [0]
        assert list(self.stim.stim_vec) == [self.base_amp]
        self.stim.reset()
        assert list(self.stim.time_vec) == []
        assert list(self.stim.stim_vec) == []

    def test_delay(self):
        """Add delay. Verify that time_vec did not change"""
        add_delay = 2.0
        l_time_vec = len(self.stim.time_vec)
        self.stim.delay(add_delay)
        assert len(self.stim.time_vec) == l_time_vec
        assert len(self.stim.stim_vec) == l_time_vec
        assert self.stim._cur_t == pytest.approx(self.base_delay + add_delay)

    def test_add_segment(self):
        """Add 2 segments"""
        A1, A2, A3 = 1.0, 2.0, 4.0
        D1, D2 = 2.0, 7.0
        self.stim.add_segment(A1, D1, A2)
        self.stim.add_segment(A3, D2)
        assert list(self.stim.stim_vec) == pytest.approx(
            [self.base_amp, A1, A2, A3, A3]
        )
        assert list(self.stim.time_vec) == pytest.approx(
            [
                0,
                self.base_delay,
                self.base_delay + D1,
                self.base_delay + D1,
                self.base_delay + D1 + D2,
            ]
        )

    def test_add_pulse_and_ramp(self):
        """Add a pulse/ramp segment and verify correct amplitude and timing"""
        A1, A2, A3, new_base_amp = 3.0, 4.0, 2.5, 5.0
        D1, D2 = 0.5, 0.8
        self.stim.add_pulse(A1, D1)
        self.stim.add_ramp(A2, A3, D2, base_amp=new_base_amp)

        assert list(self.stim.time_vec) == pytest.approx(
            [
                0.0,
                self.base_delay,
                self.base_delay,
                self.base_delay + D1,
                self.base_delay + D1,
                self.base_delay + D1,
                self.base_delay + D1,
                self.base_delay + D1 + D2,
                self.base_delay + D1 + D2,
            ]
        )
        assert list(self.stim.stim_vec) == pytest.approx(
            [
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
        )

    def test_add_train(self):
        """ Add train of pulses. Check for negative delays. """
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
        assert list(self.stim.time_vec) == pytest.approx(
            [0.0, 1.0, 1.0, 1.6, 1.6, 3.0, 3.0, 3.6, 3.6, 5.0, 5.0, 5.5, 5.5, 5.5]
        )
        assert list(self.stim.stim_vec) == pytest.approx(
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
            ]
        )

    def test_class_methods(self):
        """Test class methods"""
        assert isinstance(st.SignalSource.pulse(5.0, 10, base_amp=0.0), st.SignalSource)
        assert isinstance(
            st.SignalSource.ramp(1.0, 5.0, 10, base_amp=0.0), st.SignalSource
        )
        assert isinstance(
            st.SignalSource.train(1.0, 50, 10, 100, base_amp=0.0), st.SignalSource
        )
        assert isinstance(
            st.SignalSource.sin(1.0, 100, 50, base_amp=0.0), st.SignalSource
        )
        assert isinstance(
            st.SignalSource.noise(0.0, 1.0, 100, dt=0.5, base_amp=0.0), st.SignalSource
        )
        # assert isinstance(st.SignalSource.shot_noise(1.0, 1.0, 1.0, 1.0, 1.0,
        # 100, dt=0.25, base_amp=0.0), st.SignalSource)
        assert isinstance(
            st.SignalSource.ornstein_uhlenbeck(
                1.0, 1.0, 0.0, 100, dt=0.25, base_amp=0.0
            ),
            st.SignalSource,
        )


# class TestStimuli:
#     def setup_method(self):
#         rng = Random123(1, 2, 3)
#         self.stim = CurrentSource(rng=rng)

#     def test_flat_segment(self):
#         self.stim.add_segment(1.2, 10)
#         assert list(self.stim.time_vec) == [0, 10]
#         assert list(self.stim.stim_vec) == [1.2, 1.2]

#     def test_pulse(self):
#         self.stim.add_pulse(1.2, 10)
#         assert list(self.stim.time_vec) == [0, 0, 10, 10]
#         assert list(self.stim.stim_vec) == [0, 1.2, 1.2, 0]

#     @pytest.mark.parametrize("base_amp", [-1, 0, 1.5])
#     def test_pulse_diff_base(self, base_amp):
#         self.stim.add_pulse(1.2, 10, base_amp=base_amp)
#         assert list(self.stim.time_vec) == [0, 0, 10, 10]
#         assert list(self.stim.stim_vec) == [base_amp, 1.2, 1.2, base_amp]

#     def test_two_pulses(self):
#         self.stim.add_pulse(1.2, 10)
#         self.stim.delay(5)
#         self.stim.add_pulse(0.8, 5)
#         assert list(self.stim.time_vec) == [0, 0, 10, 10, 15, 15, 20, 20]
#         assert list(self.stim.stim_vec) == [0, 1.2, 1.2, 0, 0, 0.8, 0.8, 0]

#     def test_ramp(self):
#         self.stim.add_ramp(5, 7.5, 10)
#         assert list(self.stim.time_vec) == [0, 0, 10, 10]
#         assert list(self.stim.stim_vec) == [0, 5, 7.5, 0]

#     def test_delay_ramp(self):
#         # When a delay is specified (in ctor or factory) base_amp is set on t=0 too
#         sig = CurrentSource.ramp(1, 2, 2, base_amp=-1, delay=10)
#         assert list(sig.time_vec) == [0, 10, 10, 12, 12]
#         assert list(sig.stim_vec) == [-1, -1, 1, 2, -1]

#     def test_train(self):
#         self.stim.add_train(1.2, 10, 20, 350)
#         # At 10Hz pulses have T=100ms
#         # We end up with 4 pulses, the last one with reduced rest phase
#         assert list(self.stim.time_vec) == [0, 0, 20, 20, 100, 100, 120, 120, 200, 200, 220, 220,
#                                             300, 300, 320, 320, 350]
#         assert list(self.stim.stim_vec) == [0, 1.2, 1.2, 0] * 4 + [0]

#     def test_sin(self):
#         self.stim.add_sin(1, 0.1, 10000)
#         assert list(self.stim.time_vec) == pytest.approx([0, 0.025, 0.05, 0.075, 0.1, 0.1])
#         assert list(self.stim.stim_vec) == pytest.approx([0, 1, 0, -1, 0, 0])

#     def test_sin_long(self):
#         self.stim.add_sin(1, 200, 10, 25)
#         assert list(self.stim.time_vec) == pytest.approx([0, 25, 50, 75, 100, 125, 150,
#                                                           175, 200, 200])
#         assert list(self.stim.stim_vec) == pytest.approx([0, 1, 0, -1] * 2 + [0, 0])

#     def test_add_pulses(self):
#         self.stim.add_pulses(0.5, 1, 2, 3, 4, base_amp=0.1)
#         assert list(self.stim.time_vec) == [0, 0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2]
#         assert list(self.stim.stim_vec) == [0.1, 1, 1, 2, 2, 3, 3, 4, 4, 0.1]

#     def test_noise(self):
#         self.stim.add_noise(0.5, 0.1, 5)
#         assert list(self.stim.time_vec) == pytest.approx([0, 0, 0.5, 1, 1.5, 2, 2.5,
#                                                           3, 3.5, 4, 4.5, 5, 5])
#         assert list(self.stim.stim_vec) == pytest.approx([0, 0.70681968, 0.56316322, 0.5539058,
#                                                           0.6810689, 0.20896532, 1.00691217,
#                                                           0.78783759, 0.68817496, -6.4286609e-05,
# 0.21165959, 0.03874813, 0])

#     def test_shot_noise(self):
#         self.stim.add_shot_noise(4.0, 0.4, 2E3, 40E-3, 16E-4, 2)
#         assert list(self.stim.time_vec) == pytest.approx([0, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
#                                                           1.75, 2.0, 2.0])
#         assert list(self.stim.stim_vec) == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0700357,
# 0.1032799, 0.1170881, 0.1207344, 0.0])

#     def test_ornstein_uhlenbeck(self):
#         stim_g = ConductanceSource(rng=Random123(1, 2, 3))
#         stim_g.add_ornstein_uhlenbeck(2.8, 0.0042, 0.029, 2)
#         assert list(stim_g.time_vec) == pytest.approx([0, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
#                                                        1.75, 2.0, 2.0])
#         assert list(stim_g.stim_vec) == pytest.approx([0.0, 0.029, 0.02933925, 0.02959980,
#                                                        0.03052109, 0.02882802, 0.03156533,
# 0.03289219, 0.03357043, 0.03049419, 0.0])

#     def test_stacked(self):
#         """
#         Assert each individual stim, namely noise, starts and returns to zero
#         """
#         stim = self.stim
#         stim.add_noise(0.5, 0.1, 2)  # dt is 0.5
#         STIM1_SAMPLES = 7  # 4+1+2
#         assert stim.time_vec.size() == stim.stim_vec.size() == STIM1_SAMPLES
#         assert stim.time_vec[0] == 0.0
#         assert stim.stim_vec[0] == 0.0
#         assert stim.time_vec[-1] == 2.0
#         assert stim.stim_vec[-1] == 0.0

#         stim.delay(2)
#         stim.add_shot_noise(4.0, 0.4, 2E3, 0.5, 0.1, 2)  # dt is 0.025
#         STIM2_SAMPLES = 11  # 8+1+2
#         assert stim.time_vec.size() == stim.stim_vec.size() == STIM1_SAMPLES + STIM2_SAMPLES
#         assert stim.time_vec[STIM1_SAMPLES + 1] == 4.0
#         assert stim.stim_vec[STIM1_SAMPLES + 1] == 0.0
#         assert stim.time_vec[-1] == 6.0
#         assert stim.stim_vec[-1] == 0.0
