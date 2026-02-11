"""Stimuli sources. inc current and conductance sources which can be attached to cells"""

import logging

import numpy as np

from .random import RNG, gamma
from neurodamus.core import NeuronWrapper as Nd


class SignalSource:
    def __init__(self, base_amp=0.0, *, delay=0, rng=None, represents_physical_electrode=False):
        """Creates a new signal source, which can create composed signals
        Args:
            base_amp: The base (resting) amplitude of the signal (Default: 0)
            rng: The Random Number Generator. Used in the Noise functions
            represents_physical_electrode: Whether the source represents a phsyical
            electrode or missing synaptic input
        """
        h = Nd.h
        self.stim_vec = h.Vector()
        self.time_vec = h.Vector()
        self._cur_t = 0
        self._base_amp = base_amp
        self._rng = rng
        self._represents_physical_electrode = represents_physical_electrode
        if delay > 0.0:
            self._add_point(base_amp)
            self._cur_t = delay

    def _add_point(self, amp):
        """Appends a single point to the time-signal source.
        Note: It doesnt advance time, not supposed to be called directly
        """
        self.time_vec.append(self._cur_t)
        self.stim_vec.append(amp)

    def delay(self, duration):
        """Increments the ref time so that the next created signal is delayed"""
        # NOTE: We rely on the fact that Neuron allows "instantaneous" changes
        # and made all signal shapes return to base_amp. Therefore delay() doesn't
        # need to introduce any point to avoid interpolation.
        self._cur_t += duration
        return self

    def add_segment(self, amp, duration, amp2=None):
        """Sets a linear signal for a certain duration.

        If amp2 is None (default) then we have constant signal
        """
        self._add_point(amp)
        self.delay(duration)
        self._add_point(amp if amp2 is None else amp2)
        return self

    def add_pulse(self, max_amp, duration, **kw):
        """Add a constant-amplitude pulse.

        Generates a pulse with a constant amplitude (`max_amp`) for the specified `duration`.
        This is a special case of `add_ramp` with no amplitude change over time.
        """
        return self.add_ramp(max_amp, max_amp, duration, **kw)

    def add_ramp(self, amp1, amp2, duration, **kw):
        """Add a linear amplitude ramp.

        Creates a ramp signal that linearly changes amplitude from `amp1` to `amp2` over
        the given `duration`. All intermediate values between the start and end times
        are linearly interpolated.
        """
        base_amp = kw.get("base_amp", self._base_amp)
        self._add_point(base_amp)
        self.add_segment(amp1, duration, amp2)
        self._add_point(base_amp)
        return self

    def add_train(self, amp, frequency, pulse_duration, total_duration, **kw):
        """Stimulus with repeated pulse injections at a specified frequency.

        Args:
            amp (float): Amplitude of each pulse.
            frequency (float): Number of pulses per second (Hz).
            pulse_duration (float): Duration of a single pulse (peak time) in milliseconds.
            total_duration (float): Total duration of the pulse train in milliseconds.
            base_amp (float, optional): Base amplitude (default is 0.0).

        Returns:
            SignalSource: The instance of the SignalSource class with the configured pulse train.
        """
        base_amp = kw.get("base_amp", self._base_amp)
        tau = 1000 / frequency
        delay = tau - pulse_duration

        # we cannot have overlapping pulses otherwise we may go back in time.
        # For now it is disabled until we decide how to handle this
        if delay < 0.0:
            raise ValueError(
                f"Invalid configuration: The pulse duration ({pulse_duration} ms) is "
                f"longer than the pulse interval ({tau} ms). Calculated delay: "
                f"{delay} ms. Please adjust the pulse duration or frequency."
            )

        number_pulses = int(total_duration / tau)
        for _ in range(number_pulses):
            self.add_pulse(amp, pulse_duration, base_amp=base_amp)
            self.delay(delay)

        # Add final pulse, possibly partial
        remaining_time = total_duration - number_pulses * tau
        if pulse_duration <= remaining_time:
            self.add_pulse(amp, pulse_duration, base_amp=base_amp)
            self.delay(min(delay, remaining_time - pulse_duration))
        else:
            self.add_pulse(amp, remaining_time, base_amp=base_amp)
        # Last point
        self._add_point(base_amp)
        return self

    def add_sin(self, amp, total_duration, freq, step=0.025, **kw):
        """Builds a sinusoidal signal.

        Args:
            amp: The max amplitude of the wave
            total_duration: Total duration, in ms
            freq: The wave frequency, in Hz
            step: The step, in ms (default: 0.025)
        """
        base_amp = kw.get("base_amp", self._base_amp)

        tvec = Nd.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + total_duration, step)
        self.time_vec.append(tvec)
        self.delay(total_duration)

        stim = Nd.h.Vector(len(tvec))
        stim.sin(freq, 0.0, step)
        stim.mul(amp)
        self.stim_vec.append(stim)
        self._add_point(base_amp)  # Last point
        return self

    def add_noise(self, mean, variance, duration, dt=0.5):
        """Adds a noise component to the signal."""
        rng = self._rng or RNG()  # Creates a default RNG
        if not self._rng:
            logging.warning("Using a default RNG for noise generation")
        rng.normal(mean, variance)
        tvec = Nd.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + duration, dt)
        svec = Nd.h.Vector(len(tvec))
        svec.setrand(rng)

        # Delimit noise signals with base_amp
        # Otherwise Neuron does interpolation with surrounding points
        self._add_point(self._base_amp)
        self.time_vec.append(tvec)
        self.stim_vec.append(svec)
        self._cur_t += duration
        self._add_point(self._base_amp)
        return self

    def add_shot_noise(  # noqa: PLR0914
        self,
        tau_D,  # noqa: N803
        tau_R,  # noqa: N803
        rate,
        amp_mean,
        amp_var,
        duration,
        dt=0.25,
    ):
        """Adds a Poisson shot noise signal with gamma-distributed amplitudes and
        bi-exponential impulse response: https://paulbourke.net/miscellaneous/functions/

        tau_D: bi-exponential decay time [ms]
        tau_R: bi-exponential raise time [ms]
        rate: Poisson event rate [Hz]
        amp_mean: mean of gamma-distributed amplitudes [nA]
        amp_var: variance of gamma-distributed amplitudes [nA^2]
        duration: duration of signal [ms]
        dt: timestep [ms]
        """
        from math import exp, isclose, log, sqrt

        rng = self._rng or RNG()  # Creates a default RNG
        if not self._rng:
            logging.warning("Using a default RNG for shot noise generation")

        if isclose(tau_R, tau_D):
            raise NotImplementedError(
                f"tau_R ({tau_R}), and tau_D ({tau_D}) are too close. Edge case not implemented"
            )

        tvec = Nd.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + duration, dt)  # time vector
        ntstep = len(tvec)  # total number of timesteps

        rate_ms = rate / 1000  # rate in 1 / ms [mHz]
        napprox = 1 + int(duration * rate_ms)  # approximate number of events, at least one
        napprox = int(napprox + 3 * sqrt(napprox))  # better bound, as in elephant

        exp_scale = 1 / rate  # scale parameter of exponential distribution of time intervals
        rng.negexp(exp_scale)
        iei = Nd.h.Vector(napprox)
        iei.setrand(rng)  # generate inter-event intervals

        ev = Nd.h.Vector()
        ev.integral(iei, 1).mul(1000)  # generate events in ms

        assert ev[-1] >= duration, (
            f"The last event (ev[-1]: {ev[-1]}) is before "
            f"duration: {duration}. This should not be possible!"
        )

        ev.where("<", duration)  # remove events exceeding duration
        ev.div(dt)  # divide events by timestep

        nev = Nd.h.Vector([round(x) for x in ev])  # round to integer timestep index
        nev.where("<", ntstep)  # remove events exceeding number of timesteps

        sign = 1
        # if amplitude mean is negative, invert sign of current
        if amp_mean < 0:
            amp_mean = -amp_mean
            sign = -1

        gamma_scale = amp_var / amp_mean  # scale parameter of gamma distribution
        gamma_shape = amp_mean / gamma_scale  # shape parameter of gamma distribution
        # sample gamma-distributed amplitudes
        amp = gamma(rng, gamma_shape, gamma_scale, len(nev))

        E = Nd.h.Vector(ntstep, 0)  # full signal
        for n, A in zip(nev, amp, strict=True):
            E.x[int(n)] += sign * A  # add impulses, may overlap due to rounding to timestep

        # perform equivalent of convolution with bi-exponential impulse response
        # through a composite autoregressive process with impulse train as innovations

        # unitless quantities (time measured in timesteps)
        a = exp(-dt / tau_D)
        b = exp(-dt / tau_R)
        D = -log(a)
        R = -log(b)
        t_peak = log(R / D) / (R - D)
        A = (a / b - 1) / (a**t_peak - b**t_peak)

        P = Nd.h.Vector(ntstep, 0)
        B = Nd.h.Vector(ntstep, 0)

        # composite autoregressive process with exact solution
        # P[n] = b * (a ^ n - b ^ n) / (a - b)
        # for unit response B[0] = P[0] = 0, E[0] = 1
        for n in range(1, ntstep):
            P.x[n] = a * P[n - 1] + b * B[n - 1]
            B.x[n] = b * B[n - 1] + E[n - 1]

        P.mul(A)  # normalize to peak amplitude

        self._add_point(self._base_amp)
        self.time_vec.append(tvec)
        self.stim_vec.append(P)
        self._cur_t += duration
        self._add_point(self._base_amp)

        return self

    def add_ornstein_uhlenbeck(self, tau, sigma, mean, duration, dt=0.25):
        """Adds an Ornstein-Uhlenbeck process with given correlation time,
        standard deviation and mean value.

        tau: correlation time [ms], white noise if zero
        sigma: standard deviation [uS]
        mean: mean value [uS]
        duration: duration of signal [ms]
        dt: timestep [ms]
        """
        from math import exp, sqrt

        rng = self._rng or RNG()  # Creates a default RNG
        if not self._rng:
            logging.warning("Using a default RNG for Ornstein-Uhlenbeck process")

        tvec = Nd.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + duration, dt)  # time vector
        ntstep = len(tvec)  # total number of timesteps

        svec = Nd.h.Vector(ntstep, 0)  # stim vector

        noise = Nd.h.Vector(ntstep)  # Gaussian noise
        rng.normal(0.0, 1.0)
        noise.setrand(rng)  # generate Gaussian noise

        if tau < 1e-9:
            svec = noise.mul(sigma)  # white noise
        else:
            mu = exp(-dt / tau)  # auxiliar factor [unitless]
            A = sigma * sqrt(1 - mu * mu)  # amplitude [uS]
            noise.mul(A)  # scale noise by amplitude [uS]

            # Exact update formula (independent of dt) from Gillespie 1996
            for n in range(1, ntstep):
                svec.x[n] = svec[n - 1] * mu + noise[n]  # signal [uS]

        svec.add(mean)  # shift signal by mean value [uS]

        self._add_point(self._base_amp)
        self.time_vec.append(tvec)
        self.stim_vec.append(svec)
        self._cur_t += duration
        self._add_point(self._base_amp)

        return self

    # PLOTTING
    def plot(self, ylims=None):
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # (nrows, ncols, axnum)
        ax.plot(self.time_vec, self.stim_vec, label="Signal amplitude")
        ax.legend()
        if ylims:
            ax.set_ylim(*ylims)
        fig.show()

    # ==== Helpers =====
    # Helper methods forward generic kwargs to base class, like rng and delay

    @classmethod
    def pulse(cls, max_amp, duration, base_amp=0.0, **kw):
        return cls(base_amp, **kw).add_pulse(max_amp, duration)

    @classmethod
    def ramp(cls, amp1, amp2, duration, base_amp=0.0, **kw):
        return cls(base_amp, **kw).add_ramp(amp1, amp2, duration)

    @classmethod
    def train(cls, amp, frequency, pulse_duration, total_duration, base_amp=0.0, **kw):
        return cls(base_amp, **kw).add_train(amp, frequency, pulse_duration, total_duration)

    @classmethod
    def sin(cls, amp, total_duration, freq, step=0.025, base_amp=0.0, **kw):
        return cls(base_amp, **kw).add_sin(amp, total_duration, freq, step)

    @classmethod
    def noise(cls, mean, variance, duration, dt=0.5, base_amp=0.0, **kw):
        return cls(base_amp, **kw).add_noise(mean, variance, duration, dt)

    @classmethod
    def shot_noise(cls, tau_D, tau_R, rate, amp_mean, var, duration, dt=0.25, base_amp=0.0, **kw):  # noqa: N803
        return cls(base_amp, **kw).add_shot_noise(tau_D, tau_R, rate, amp_mean, var, duration, dt)

    @classmethod
    def ornstein_uhlenbeck(cls, tau, sigma, mean, duration, dt=0.25, base_amp=0.0, **kw):
        return cls(base_amp, **kw).add_ornstein_uhlenbeck(tau, sigma, mean, duration, dt)


class CurrentSource(SignalSource):
    _all_sources = []

    def __init__(self, base_amp=0.0, *, delay=0, rng=None, represents_physical_electrode=False):
        """Creates a new current source that injects a signal under IClamp"""
        super().__init__(
            base_amp,
            delay=delay,
            rng=rng,
            represents_physical_electrode=represents_physical_electrode,
        )
        self._clamps = set()
        self._all_sources.append(self)

    class _Clamp:
        def __init__(
            self,
            cell_section,
            position=0.5,
            clamp_container=None,
            stim_vec_mode=True,
            time_vec=None,
            stim_vec=None,
            represents_physical_electrode=False,
            **clamp_params,
        ):
            # Checks if source does not represent physical electrode,
            # otherwise fall back to IClamp.
            self.clamp = (
                Nd.h.IClamp(position, sec=cell_section)
                if represents_physical_electrode
                else Nd.h.MembraneCurrentSource(position, sec=cell_section)
            )

            if stim_vec_mode:
                assert time_vec is not None
                assert stim_vec is not None
                self.clamp.dur = time_vec[-1]
                stim_vec.play(self.clamp._ref_amp, time_vec, 1)
            else:
                # this is probably unused
                for param, val in clamp_params.items():
                    setattr(self.clamp, param, val)

            # Clamps must be kept otherwise they are garbage-collected
            self._all_clamps = clamp_container
            clamp_container.add(self)

        def detach(self):
            """Detaches a clamp from a cell, destroying it"""
            self._all_clamps.discard(self)
            del self.clamp  # Force del on the clamp (there might be references to self)

    def attach_to(self, section, position=0.5):
        return CurrentSource._Clamp(
            section,
            position,
            self._clamps,
            stim_vec_mode=True,
            time_vec=self.time_vec,
            stim_vec=self.stim_vec,
            represents_physical_electrode=self._represents_physical_electrode,
        )


class ConductanceSource(SignalSource):
    _all_sources = []

    def __init__(self, reversal=0.0, *, delay=0.0, rng=None, represents_physical_electrode=False):
        """Creates a new conductance source that injects a conductance by driving
        the rs of an SEClamp at a given reversal potential.

        reversal: reversal potential of conductance (mV)
        """
        # set SignalSource's base_amp to zero
        super().__init__(
            reversal,
            delay=delay,
            rng=rng,
            represents_physical_electrode=represents_physical_electrode,
        )
        self._reversal = reversal  # set reversal from base_amp parameter in classmethods
        self._clamps = set()
        self._all_sources.append(self)

    class _DynamicClamp:
        def __init__(
            self,
            cell_section,
            position=0.5,
            clamp_container=None,
            time_vec=None,
            stim_vec=None,
            reversal=0.0,
            represents_physical_electrode=False,
        ):
            # source does not represent physical electrode,
            # otherwise fall back to SEClamp.
            self.clamp = (
                Nd.h.SEClamp(position, sec=cell_section)
                if represents_physical_electrode
                else Nd.h.ConductanceSource(position, sec=cell_section)
            )

            assert time_vec is not None
            assert stim_vec is not None
            self.clamp.dur1 = time_vec[-1]
            self.clamp.amp1 = reversal
            # support delay with initial zero
            self.time_vec = Nd.h.Vector(1, 0).append(time_vec)
            self.stim_vec = Nd.h.Vector(1, 0).append(stim_vec)
            # replace self.stim_vec with inverted and clamped signal
            # rs is in MOhm, so conductance is in uS (micro Siemens)
            self.stim_vec = Nd.h.Vector(
                [1 / x if abs(x) > 1e-9 else (1e9 if x >= 0 else -1e9) for x in self.stim_vec]
            )
            self.stim_vec.play(self.clamp._ref_rs, self.time_vec, 1)
            # Clamps must be kept otherwise they are garbage-collected
            self._all_clamps = clamp_container
            clamp_container.add(self)

        def detach(self):
            """Detaches a clamp from a cell, destroying it"""
            self._all_clamps.discard(self)
            del self.clamp  # Force del on the clamp (there might be references to self)

    def attach_to(self, section, position=0.5):
        return ConductanceSource._DynamicClamp(
            section,
            position,
            self._clamps,
            self.time_vec,
            self.stim_vec,
            self._reversal,
            represents_physical_electrode=self._represents_physical_electrode,
        )


class ElectrodeSource:
    """Constructs an extracellular potential field as the sum of multiple user-defined e-fields,
    and applies the resulting signal to the segment's e_extracellular.

    Args:
        base_amp: baseline amplitude when signal is inactive
        delay: start time delay in ms
        duration: duration of the signal, not including ramp up and ramp down
        fields: list of user-defined electric field components (e.g. cosinuoid fields)
        duration: duration of the signal, not including ramp up and ramp down.
        ramp_up_time: duration during which the signal amplitude ramps up linearly from 0, in ms
        ramp_down_time: duration during which the signal amplitude ramps down linearly to 0, in ms
    """

    def __init__(self, base_amp, delay, duration, fields, ramp_up_time, ramp_down_time, dt):
        self.time_vec = Nd.h.Vector()  # Time points for stimulus waveform
        self._cur_t = 0
        self._base_amp = base_amp
        self._delay = delay
        self.fields = fields
        self.duration = duration
        self.dt = dt
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        # for delay, add cur_t as the first point, then advance cur_t
        if delay > 0:
            self.time_vec.append(self._cur_t)
            self._cur_t = delay
        self.efields = self.add_cosines()  # np.array, E_x, E_y, E_z vectors varied by time
        self.segment_displacements = {}  # {segment: displacement vectors in x/y/z w.r.t ground}
        self.segment_potentials = []  # potentials that are applied to segment.extracellular._ref_e

    def delay(self, duration):
        """Increments the ref time so that the next created signal is delayed"""
        # NOTE: We rely on the fact that Neuron allows "instantaneous" changes
        # and made all signal shapes return to base_amp. Therefore delay() doesn't
        # need to introduce any point to avoid interpolation.
        self._cur_t += duration
        return self

    def add_cosines(self):
        """Add multiple cosinusoidal signals
        Returns: a list of cosine signal vectors
        """
        total_duration = self.duration + self.ramp_up_time + self.ramp_down_time
        tvec = Nd.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + total_duration, self.dt)
        self.time_vec.append(tvec)
        self.delay(total_duration)
        # add last time point: the next step after duration to revert signal back to base_amp
        self.time_vec.append(self._cur_t + self.dt)
        res_x = Nd.h.Vector(len(self.time_vec))
        res_y = Nd.h.Vector(len(self.time_vec))
        res_z = Nd.h.Vector(len(self.time_vec))
        for field in self.fields:
            vec = Nd.h.Vector(len(tvec))
            freq = field.get("Frequency", 0)
            phase = field.get("Phase", 0)
            Ex = field["Ex"]
            Ey = field["Ey"]
            Ez = field["Ez"]
            vec.sin(freq, phase + np.pi / 2, self.dt)
            self.apply_ramp(vec, self.dt)
            if self._delay > 0:
                # for delay, insert base_amp for the 1st point
                vec.insrt(0, self._base_amp)
            vec.append(self._base_amp)  # add last point
            res_x.add(vec.c().mul(Ex))
            res_y.add(vec.c().mul(Ey))
            res_z.add(vec.c().mul(Ez))

        return np.array([res_x, res_y, res_z])

    def compute_potentials(self, displacement_vec):
        return np.dot(displacement_vec, self.efields) * 1e3  # Converts from V to mV

    def apply_ramp(self, signal_vec, step):
        """Apply signal ramp up and down
        Args:
            signal_vec: the signal vector to apply ramp, type hoc Vector
            step: time step
        """
        ramp_up_number = int(
            self.ramp_up_time / step
        )  # Number of time points during the ramp-up window
        ramp_down_number = int(
            self.ramp_down_time / step
        )  # Number of time points during the ramp-down window

        if ramp_up_number > 0:
            ramp_up = np.linspace(0, 1, ramp_up_number)
            signal_vec[:ramp_up_number] *= ramp_up
        if ramp_down_number > 0:
            ramp_down = np.linspace(1, 0, ramp_down_number)
            signal_vec[-ramp_down_number:] *= ramp_down

    def apply_segment_potentials(self):
        """Apply potentials to segment.extracellular._ref_e"""
        for segment, displacement in self.segment_displacements.items():
            section = segment.sec
            e_ext_vec = Nd.h.Vector(self.compute_potentials(displacement))
            if not section.has_membrane("extracellular"):
                section.insert("extracellular")
            # Neuron vector play without interpolation, continuous=0
            e_ext_vec.play(segment.extracellular._ref_e, self.time_vec, 0)
            self.segment_potentials.append(e_ext_vec)

        self.cleanup()

    def cleanup(self):
        """Clear unused list variable to free memory"""
        self.efields = None
        self.segment_displacements = None

    def __iadd__(self, other):
        """Combined with another ElectrodeSource object
        1. combine the time_vec
        2. combine efields E_x/y/z, if the time overlaps, should be summed
        """
        assert np.isclose(self.dt, other.dt), "multiple extracellular stimuli must have common dt"
        combined_time_vec, self.efields = self._combine_time_efields(
            self.time_vec.as_numpy(),
            self.efields,
            other.time_vec.as_numpy(),
            other.efields,
            self._delay > 0,
            other._delay > 0,
            self.dt,
        )
        self.time_vec = Nd.h.Vector(combined_time_vec)
        return self

    @staticmethod
    def _combine_time_efields(t1_vec, efields1, t2_vec, efields2, is_delay1, is_delay2, dt):
        """Combine time and efields vectors from 2 ElectrodeSource objects.
        In case of delay, the 1st element of the time-efields vectors should be removed,
        and the delay time is always divisible by dt

        Args:
            t1_vec, t2_vec : numpy arrays of size n_timepoints, always ordered
            efields1, efields2: shape (3, n_timepoints) for Ex, Ey, Ez
            is_delay1 : if the stimulus 1 has delay
            is_delay2 : if the stimulus 2 has delay
        Returns: np.array, the combined time and efields vectors
        """
        if is_delay1:
            t1_vec = t1_vec[1:]
            if not (np.isclose(t1_vec[0] % dt, 0) or np.isclose(t1_vec[0] % dt, dt)):
                raise ValueError(
                    f"ElectrodeSource time vector must be divisible by dt {dt}, "
                    f"check the delay parameter {t1_vec[0]}"
                )
            efields1 = efields1[:, 1:]  # Remove first column for all 3 rows
        if is_delay2:
            t2_vec = t2_vec[1:]
            if not (np.isclose(t2_vec[0] % dt, 0) or np.isclose(t2_vec[0] % dt, dt)):
                raise ValueError(
                    f"ElectrodeSource time vector must be divisible by dt {dt}, "
                    f"check the delay parameter {t2_vec[0]}"
                )
            efields2 = efields2[:, 1:]

        # Convert time -> integer ticks
        t1_ticks = np.round(t1_vec / dt).astype(np.int64)
        t2_ticks = np.round(t2_vec / dt).astype(np.int64)

        if not (t1_ticks[-1] < t2_ticks[0] or t2_ticks[-1] < t1_ticks[0]):
            # Range overlap or exact continuous, exact union of time_ticks
            combined_time_ticks = np.union1d(t1_ticks, t2_ticks)
            # Stepwise amplitude lookup (vectorized)
            idx1_left = (
                np.searchsorted(t1_ticks, combined_time_ticks, side="right") - 1
            )  # idx=-1 for not existing points smaller than t1_ticks
            idx1_right = np.searchsorted(
                t1_ticks, combined_time_ticks, side="left"
            )  # idx=len(vec) for not existing points larger than t1_ticks
            mask1 = idx1_left == idx1_right  # find the common existing points
            idx2_left = np.searchsorted(t2_ticks, combined_time_ticks, side="right") - 1
            idx2_right = np.searchsorted(t2_ticks, combined_time_ticks, side="left")
            mask2 = idx2_left == idx2_right

            combined_efields = np.zeros((len(efields1), len(combined_time_ticks)), dtype=float)
            combined_efields[:, mask1] += efields1[:, idx1_left[mask1]]
            combined_efields[:, mask2] += efields2[:, idx2_left[mask2]]

        # Non-overlapping: concatenate
        elif t1_ticks[-1] < t2_ticks[0]:
            combined_time_ticks = np.concatenate((t1_ticks, t2_ticks))
            combined_efields = np.concatenate((efields1, efields2), axis=1)
        else:
            combined_time_ticks = np.concatenate((t2_ticks, t1_ticks))
            combined_efields = np.concatenate((efields2, efields1), axis=1)

        # Convert ticks -> float time
        combined_time_vec = combined_time_ticks.astype(float) * dt

        if combined_time_vec[0] > 0:
            # in case of delay add back t=0 stim=0
            combined_time_vec = np.concatenate([[0.0], combined_time_vec])
            combined_efields = np.concatenate([np.zeros((combined_efields.shape[0], 1)), combined_efields], axis=1)

        assert combined_efields.shape[1] == len(combined_time_vec), "Time and efield length mismatch"

        return combined_time_vec, combined_efields
