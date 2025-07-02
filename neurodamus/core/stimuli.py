"""Stimuli sources. inc current and conductance sources which can be attached to cells"""

import logging

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
        for n, A in zip(nev, amp):
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


# EStim class is a derivative of TStim for stimuli with an extracelular electrode. The main
# difference is that it collects all elementary stimuli pulses and converts them using a
# VirtualElectrode object before it injects anything
#
# The stimulus is defined on the hoc level by using the addpoint function for every (step) change
# in extracellular electrode voltage. At this stage only step changes can be used. Gradual,
# i.e. sinusoidal changes will be implemented in the future
# After every step has been defined, you have to call initElec() to perform the frequency dependent
# transformation. This transformation turns e_electrode into e_extracellular at distance d=1 micron
# from the electrode. After the transformation is complete, NO MORE STEPS CAN BE ADDED!
# You can then use inject() to first scale e_extracellular down by a distance dependent factor
# and then vector.play() it into the currently accessed compartment
#
# TODO: 1. more stimulus primitives than step. 2. a dt of 0.1 ms is hardcoded. make this flexible!
