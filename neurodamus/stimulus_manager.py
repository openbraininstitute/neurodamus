# https://bbpteam.epfl.ch/project/spaces/display/BGLIB/Neurodamus
# Copyright 2005-2021 Blue Brain Project, EPFL. All rights reserved.
"""Implements coupling artificial stimulus into simulation

New Stimulus classes must be registered, using the appropriate decorator.
Also, when instantiated by the framework, __init__ is passed three arguments
(1) target (2) stim_info: dict (3) cell_manager. Example

>>> @StimulusManager.register_type
>>> class ShotNoise:
>>>
>>> def __init__(self, target, stim_info: dict, cell_manager):
>>>     tpoints = target.getPointList(cell_manager)
>>>     for point in tpoints:
>>>         gid = point.gid
>>>         cell = cell_manager.get_cell(gid)

"""

import logging

from .core import NeuronWrapper as Nd, random
from .core.configuration import ConfigurationError, SimConfig
from .core.stimuli import ConductanceSource, CurrentSource
from .utils.logging import log_verbose


class StimulusManager:
    """A manager for synaptic artificial Stimulus.
    Old stimulus resort to hoc implementation
    """

    _stim_types = {}  # stimulus handled in Python

    def __init__(self, target_manager):
        self._target_manager = target_manager
        self._stim_seed = SimConfig.run_conf.get("StimulusSeed")
        self._stimulus = []
        self.reset_helpers()  # reset helpers for multi-cycle builds

    def interpret(self, target_spec, stim_info):
        stim_t = self._stim_types.get(stim_info["Pattern"])
        if not stim_t:
            msg = f"No implementation for Stimulus {stim_info['Pattern']}"
            raise ConfigurationError(msg)
        if self._stim_seed is None and getattr(stim_t, "IsNoise", False):
            logging.warning(
                "StimulusSeed unset (default %d), set explicitly to vary noisy stimuli across runs",
                SimConfig.rng_info.getStimulusSeed(),
            )
        target = self._target_manager.get_target(target_spec)
        log_verbose("Interpret stimulus")
        cell_manager = self._target_manager._cell_manager
        stim = stim_t(target, stim_info, cell_manager)
        self._stimulus.append(stim)

    @staticmethod
    def reset_helpers():
        ShotNoise.stim_count = 0
        Noise.stim_count = 0
        OrnsteinUhlenbeck.stim_count = 0

    @classmethod
    def register_type(cls, stim_class):
        """Registers a new class as a handler for a new stim type"""
        cls._stim_types[stim_class.__name__] = stim_class
        return stim_class

    def saveStatePreparation(self, ss_obj):
        for stim in self._stimulus:
            ss_obj.ignore(stim)


class BaseStim:
    """Barebones stimulus class"""

    IsNoise = False

    def __init__(self, _target, stim_info: dict, _cell_manager):
        self.duration = float(stim_info["Duration"])  # duration [ms]
        self.delay = float(stim_info["Delay"])  # start time [ms]
        self.represents_physical_electrode = stim_info.get("RepresentsPhysicalElectrode", False)


@StimulusManager.register_type
class OrnsteinUhlenbeck(BaseStim):
    """Ornstein-Uhlenbeck process, injected as current or conductance"""

    IsNoise = True
    stim_count = 0  # global count for seeding

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

        self.stimList = []  # sources go here

        if not self.parse_check_all_parameters(stim_info):
            return  # nothing to do, stim is a no-op

        # setup random seeds
        seed1 = OrnsteinUhlenbeck.stim_count + 2997  # stimulus block seed
        seed2 = SimConfig.rng_info.getStimulusSeed() + 291204  # stimulus type seed
        seed3 = (lambda x: x + 123) if self.seed is None else (lambda _x: self.seed)  # GID seed

        # apply stim to each point in target
        tpoints = target.getPointList(cell_manager)
        for tpoint_list in tpoints:
            gid = tpoint_list.gid
            cell = cell_manager.get_cell(gid)

            self.compute_parameters(cell)

            for sec_id, sc in enumerate(tpoint_list.sclst):
                # skip sections not in this split
                if not sc.exists():
                    continue

                rng = random.Random123(seed1, seed2, seed3(gid))  # setup RNG
                ou_args = (self.tau, self.sigma, self.mean, self.duration)
                ou_kwargs = {
                    "dt": self.dt,
                    "delay": self.delay,
                    "rng": rng,
                    "represents_physical_electrode": self.represents_physical_electrode,
                }
                # inject Ornstein-Uhlenbeck signal
                if stim_info["Mode"] == "Conductance":
                    cs = ConductanceSource.ornstein_uhlenbeck(
                        *ou_args, **ou_kwargs, base_amp=self.reversal
                    )
                else:
                    cs = CurrentSource.ornstein_uhlenbeck(*ou_args, **ou_kwargs)
                # attach source to section
                cs.attach_to(sc.sec, tpoint_list.x[sec_id])
                self.stimList.append(cs)  # save source

        OrnsteinUhlenbeck.stim_count += 1  # increment global count

    def parse_check_all_parameters(self, stim_info: dict):
        self.dt = float(stim_info.get("Dt", 0.25))  # stimulus timestep [ms]
        if self.dt <= 0:
            raise Exception(f"{self.__class__.__name__} time-step must be positive")

        self.reversal = float(stim_info.get("Reversal", 0.0))  # reversal potential [mV]

        if stim_info["Mode"] not in {"Current", "Conductance"}:
            raise Exception(
                f"{self.__class__.__name__} must be used with mode Current or Conductance"
            )

        self.tau = float(stim_info["Tau"])  # relaxation time [ms]
        if self.tau < 0:
            raise Exception(f"{self.__class__.__name__} relaxation time must be non-negative")

        # parse and check stimulus-specific parameters
        if not self.parse_check_stim_parameters(stim_info):
            return False  # nothing to do, stim is a no-op

        self.seed = stim_info.get("Seed")  # random seed override
        if self.seed is not None:
            self.seed = int(self.seed)

        return True

    def parse_check_stim_parameters(self, stim_info):
        self.sigma = float(stim_info["Sigma"])  # signal stdev [uS]
        if self.sigma <= 0:
            raise Exception(f"{self.__class__.__name__} standard deviation must be positive")

        self.mean = float(stim_info["Mean"])  # signal mean [uS]
        if self.mean < 0 and abs(self.mean) > 2 * self.sigma:
            logging.warning("%s signal is mostly zero", self.__class__.__name__)

        return True

    def compute_parameters(self, cell):
        # nothing to do
        pass


@StimulusManager.register_type
class RelativeOrnsteinUhlenbeck(OrnsteinUhlenbeck):
    """Ornstein-Uhlenbeck process, injected as current or conductance,
    relative to cell threshold current or inverse input resistance
    """

    IsNoise = True

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

    def parse_check_stim_parameters(self, stim_info):
        self.mean_perc = float(stim_info["MeanPercent"])
        self.sigma_perc = float(stim_info["SDPercent"])

        if stim_info["Mode"] == "Current":
            self.get_relative = lambda x: x.getThreshold()
        else:
            self.get_relative = lambda x: 1.0 / x.input_resistance

        return True

    def compute_parameters(self, cell):
        # threshold current [nA] or inverse input resistance [uS]
        rel_prop = self.get_relative(cell)

        self.sigma = (self.sigma_perc / 100) * rel_prop  # signal stdev [nA or uS]
        if self.sigma <= 0:
            raise Exception(f"{self.__class__.__name__} standard deviation must be positive")

        self.mean = (self.mean_perc / 100) * rel_prop  # signal mean [nA or uS]
        if self.mean < 0 and abs(self.mean) > 2 * self.sigma:
            logging.warning("%s signal is mostly zero", self.__class__.__name__)

        return True


@StimulusManager.register_type
class ShotNoise(BaseStim):
    """ShotNoise stimulus handler implementing Poisson shot noise
    with bi-exponential response and gamma-distributed amplitudes
    """

    IsNoise = True
    stim_count = 0  # global count for seeding

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

        self.stimList = []  # CurrentSource's go here

        if not self.parse_check_all_parameters(stim_info):
            return  # nothing to do, stim is a no-op

        # setup random seeds
        seed1 = ShotNoise.stim_count + 2997  # stimulus block seed
        seed2 = SimConfig.rng_info.getStimulusSeed() + 19216  # stimulus type seed
        seed3 = (lambda x: x + 123) if self.seed is None else (lambda _x: self.seed)  # GID seed

        # apply stim to each point in target
        tpoints = target.getPointList(cell_manager)
        for tpoint_list in tpoints:
            gid = tpoint_list.gid
            cell = cell_manager.get_cell(gid)

            self.compute_parameters(cell)

            for sec_id, sc in enumerate(tpoint_list.sclst):
                # skip sections not in this split
                if not sc.exists():
                    continue

                rng = random.Random123(seed1, seed2, seed3(gid))  # setup RNG
                shotnoise_args = (
                    self.tau_D,
                    self.tau_R,
                    self.rate,
                    self.amp_mean,
                    self.amp_var,
                    self.duration,
                )
                shotnoise_kwargs = {
                    "dt": self.dt,
                    "delay": self.delay,
                    "rng": rng,
                    "represents_physical_electrode": self.represents_physical_electrode,
                }
                # generate shot noise current source
                if stim_info["Mode"] == "Conductance":
                    cs = ConductanceSource.shot_noise(
                        *shotnoise_args, **shotnoise_kwargs, base_amp=self.reversal
                    )
                else:
                    cs = CurrentSource.shot_noise(*shotnoise_args, **shotnoise_kwargs)
                # attach current source to section
                cs.attach_to(sc.sec, tpoint_list.x[sec_id])
                self.stimList.append(cs)  # save CurrentSource

        ShotNoise.stim_count += 1  # increment global count

    def parse_check_all_parameters(self, stim_info: dict):
        if stim_info["Mode"] not in {"Current", "Conductance"}:
            raise Exception(
                f"{self.__class__.__name__} must be used with mode Current or Conductance"
            )

        self.reversal = float(stim_info.get("Reversal", 0.0))  # reversal potential [mV]

        # time parameters
        self.dt = float(stim_info.get("Dt", 0.25))  # stimulus timestep [ms]
        if self.dt <= 0:
            raise Exception(f"{self.__class__.__name__} time-step must be positive")

        ntstep = int(self.duration / self.dt)  # number of timesteps [1]
        if ntstep == 0:
            return False  # nothing to do, stim is a no-op

        # bi-exponential parameters
        self.tau_R = float(stim_info["RiseTime"])  # rise time [ms]
        self.tau_D = float(stim_info["DecayTime"])  # decay time [ms]
        if self.tau_R >= self.tau_D:
            klass = self.__class__.__name__
            raise Exception(f"{klass} bi-exponential rise time must be smaller than decay time")

        # parse and check stimulus-specific parameters
        if not self.parse_check_stim_parameters(stim_info):
            return False  # nothing to do, stim is a no-op

        # seed
        self.seed = stim_info.get("Seed")  # random seed override
        if self.seed is not None:
            self.seed = int(self.seed)

        return True

    def parse_check_stim_parameters(self, stim_info: dict):
        """Parse parameters for ShotNoise stimulus"""
        # event rate of Poisson process [Hz]
        self.rate = float(stim_info["Rate"])

        # mean amplitude of shots [nA or uS]
        # when negative we invert the sign of the current
        self.amp_mean = float(stim_info["AmpMean"])
        if self.amp_mean == 0:
            raise Exception(f"{self.__class__.__name__} amplitude mean must be non-zero")

        # variance of amplitude of shots [nA^2 or uS^2]
        self.amp_var = float(stim_info["AmpVar"])
        if self.amp_var <= 0:
            raise Exception(f"{self.__class__.__name__} amplitude variance must be positive")

        return self.rate > 0  # no-op if rate == 0

    def compute_parameters(self, cell):
        # nothing to do
        pass

    def params_from_mean_sd(self, mean, sd):
        """Compute bi-exponential shot noise parameters from desired mean and std. dev. of signal.

        Analytical result derived from a generalization of Campbell's theorem present in
        Rice, S.O., "Mathematical Analysis of Random Noise", BSTJ 23, 3 Jul 1944.
        """
        from math import exp, log

        # bi-exponential time to peak [ms]
        t_peak = log(self.tau_D / self.tau_R) / (1 / self.tau_R - 1 / self.tau_D)
        # bi-exponential peak height [1]
        F_peak = exp(-t_peak / self.tau_D) - exp(-t_peak / self.tau_R)

        # utility constants
        Xi = (self.tau_D - self.tau_R) / F_peak
        A = 1 / (self.tau_D + self.tau_R)
        B = 1 / ((self.tau_D + 2 * self.tau_R) * (2 * self.tau_D + self.tau_R))

        # skewness
        skew_bnd_min = (8 / 3) * (B / A**2) * (sd / mean)
        skew = (1 + self.rel_skew) * skew_bnd_min
        if skew < skew_bnd_min or skew > 2 * skew_bnd_min:
            raise Exception(f"{self.__class__.__name__} skewness out of bounds")

        # cumulants
        lambda2_1 = sd**2 / mean  # lambda2 over lambda1
        lambda3_2 = sd * skew  # lambda3 over lambda2
        theta1pk = 2 / (A * Xi) * lambda2_1  # = (1 + k) * theta
        theta2pk = (3 * A) / (4 * B * Xi) * lambda3_2  # = (2 + k) * theta

        # derived parameters
        self.amp_mean = 2 * theta1pk - theta2pk  # mean amplitude [nA or uS]
        self.amp_var = self.amp_mean * (theta2pk - theta1pk)  # variance of amplitude [nA^2 or uS^2]
        rate_ms = mean / (self.amp_mean * Xi)  # event rate in 1 / ms
        self.rate = rate_ms * 1000  # event rate in 1 / s [Hz]


@StimulusManager.register_type
class RelativeShotNoise(ShotNoise):
    """RelativeShotNoise stimulus handler, same as ShotNoise
    but parameters relative to cell threshold current or inverse input resistance
    """

    IsNoise = True

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

    def parse_check_stim_parameters(self, stim_info: dict):
        """Parse parameters for RelativeShotNoise stimulus"""
        # signal mean as percent of cell's threshold [1],
        # when negative we invert the sign of the current
        self.mean_perc = float(stim_info["MeanPercent"])

        # signal standard deviation as percent of cell's threshold [1]
        self.sd_perc = float(stim_info["SDPercent"])
        if self.sd_perc <= 0:
            raise Exception(f"{self.__class__.__name__} stdev percent must be positive")
        if self.sd_perc < 1:
            logging.warning(
                "%s stdev percent too small gives a very high event rate", self.__class__.__name__
            )

        # relative skewness of signal as a [0,1] fraction [1]
        self.rel_skew = float(stim_info.get("RelativeSkew", 0.5))
        if self.rel_skew < 0.0 or self.rel_skew > 1.0:
            raise Exception(f"{self.__class__.__name__} relative skewness must be in [0,1]")

        if stim_info["Mode"] == "Current":
            self.get_relative = lambda x: x.getThreshold()
        else:
            self.get_relative = lambda x: 1.0 / x.input_resistance

        return self.mean_perc != 0  # no-op if mean_perc == 0

    def compute_parameters(self, cell):
        # threshold current [nA] or inverse input resistance [uS]
        rel_prop = self.get_relative(cell)
        mean = (self.mean_perc / 100) * rel_prop  # desired mean [nA or uS]
        sd = (self.sd_perc / 100) * rel_prop  # desired standard deviation [nA or uS]
        super().params_from_mean_sd(mean, sd)


@StimulusManager.register_type
class AbsoluteShotNoise(ShotNoise):
    """AbsoluteShotNoise stimulus handler, same as ShotNoise
    but parameters from given mean and std. dev.
    """

    IsNoise = True

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

    def parse_check_stim_parameters(self, stim_info: dict):
        """Parse parameters for AbsoluteShotNoise stimulus"""
        # signal mean [nA]
        self.mean = float(stim_info["Mean"])

        # signal standard deviation [nA]
        self.sd = float(stim_info["Sigma"])
        if self.sd <= 0:
            raise Exception(f"{self.__class__.__name__} stdev must be positive")

        # relative skewness of signal as a [0,1] fraction [1]
        self.rel_skew = float(stim_info.get("RelativeSkew", 0.5))
        if self.rel_skew < 0.0 or self.rel_skew > 1.0:
            raise Exception(f"{self.__class__.__name__} relative skewness must be in [0,1]")

        return True

    def compute_parameters(self, cell):
        super().params_from_mean_sd(self.mean, self.sd)


@StimulusManager.register_type
class Linear(BaseStim):
    """Injects a linear current ramp."""

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

        self.stimList = []  # CurrentSource's go here

        if not self.parse_check_all_parameters(stim_info):
            return  # nothing to do, stim is a no-op

        # apply stim to each point in target
        tpoints = target.getPointList(cell_manager)
        for tpoint_list in tpoints:
            gid = tpoint_list.gid
            cell = cell_manager.get_cell(gid)

            self.compute_parameters(cell)

            for sec_id, sc in enumerate(tpoint_list.sclst):
                # skip sections not in this split
                if not sc.exists():
                    continue

                # generate ramp current source
                cs = CurrentSource.ramp(
                    self.amp_start,
                    self.amp_end,
                    self.duration,
                    delay=self.delay,
                    represents_physical_electrode=self.represents_physical_electrode,
                )
                # attach current source to section
                cs.attach_to(sc.sec, tpoint_list.x[sec_id])
                self.stimList.append(cs)  # save CurrentSource

    def parse_check_all_parameters(self, stim_info: dict):
        # Amplitude at start
        self.amp_start = float(stim_info["AmpStart"])

        # Amplitude at end (optional, else same as start)
        self.amp_end = float(stim_info.get("AmpEnd", self.amp_start))

        return self.amp_start != 0 or self.amp_end != 0  # no-op if both 0

    def compute_parameters(self, cell):
        pass  # nothing to do


@StimulusManager.register_type
class Hyperpolarizing(Linear):
    """Injects a constant step with a cell's hyperpolarizing current."""

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

    @staticmethod
    def parse_check_all_parameters(_stim_info: dict):
        return True

    def compute_parameters(self, cell):
        hypamp = cell.getHypAmp()
        self.amp_start = hypamp
        self.amp_end = hypamp


@StimulusManager.register_type
class RelativeLinear(Linear):
    """Injects a linear current ramp relative to cell threshold."""

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

    def parse_check_all_parameters(self, stim_info: dict):
        # Amplitude at start as percent of threshold
        self.perc_start = float(stim_info["PercentStart"])

        # Amplitude at end as percent of threshold (optional, else same as start)
        self.perc_end = float(stim_info.get("PercentEnd", self.perc_start))

        return self.perc_start != 0 or self.perc_end != 0  # no-op if both 0

    def compute_parameters(self, cell):
        threshold = cell.getThreshold()
        # here we use parentheses to match HOC exactly
        self.amp_start = threshold * (self.perc_start / 100)
        self.amp_end = threshold * (self.perc_end / 100)


@StimulusManager.register_type
class SubThreshold(Linear):
    """Injects a current step at some percent below a cell's threshold."""

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

    def parse_check_all_parameters(self, stim_info: dict):
        # amplitude as percent below threshold = 100%
        self.perc_less = float(stim_info["PercentLess"])

        return True

    def compute_parameters(self, cell):
        threshold = cell.getThreshold()
        # here we use parentheses to match HOC exactly
        self.amp_start = threshold * (100 - self.perc_less) / 100
        self.amp_end = self.amp_start


@StimulusManager.register_type
class Noise(BaseStim):
    """Inject a noisy (gaussian) current step, relative to cell threshold or not."""

    IsNoise = True
    stim_count = 0  # global count for seeding

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

        self.stimList = []  # CurrentSource's go here

        self.parse_check_all_parameters(stim_info)

        sim_dt = float(SimConfig.run_conf["Dt"])  # simulation time-step [ms]

        # apply stim to each point in target
        tpoints = target.getPointList(cell_manager)
        for tpoint_list in tpoints:
            gid = tpoint_list.gid
            cell = cell_manager.get_cell(gid)

            self.compute_parameters(cell)

            rng = random.Random123(
                Noise.stim_count + 100, SimConfig.rng_info.getStimulusSeed() + 500, gid + 300
            )

            # draw already used numbers
            if self.delay > 0:
                self.draw_already_used_numbers(rng, sim_dt)

            for sec_id, sc in enumerate(tpoint_list.sclst):
                # skip sections not in this split
                if not sc.exists():
                    continue

                # generate noise current source
                cs = CurrentSource.noise(
                    self.mean,
                    self.var,
                    self.duration,
                    dt=self.dt,
                    delay=self.delay,
                    rng=rng,
                    represents_physical_electrode=self.represents_physical_electrode,
                )
                # attach current source to section
                cs.attach_to(sc.sec, tpoint_list.x[sec_id])
                self.stimList.append(cs)  # save CurrentSource

        Noise.stim_count += 1  # increment global count

    def parse_check_all_parameters(self, stim_info: dict):
        self.dt = float(stim_info.get("Dt", 0.5))  # stimulus timestep [ms]
        if self.dt <= 0:
            raise Exception(f"{self.__class__.__name__} time-step must be positive")

        if "Mean" in stim_info:
            self.is_relative = False
            self.mean = float(stim_info["Mean"])  # noise current mean [nA]

            self.var = float(stim_info["Variance"])  # noise current variance [nA]
            if self.var <= 0:
                raise Exception(f"{self.__class__.__name__} variance must be positive")
        else:
            self.is_relative = True
            # noise current mean as percent of threshold
            self.mean_perc = float(stim_info["MeanPercent"])

            # noise current variance as percent of threshold
            self.var_perc = float(stim_info["Variance"])
            if self.var_perc <= 0:
                raise Exception(f"{self.__class__.__name__} variance percent must be positive")

        return True

    def compute_parameters(self, cell):
        if self.is_relative:
            threshold = cell.getThreshold()  # threshold current [nA]
            # here threshold MUST be first factor to match HOC exactly
            self.mean = threshold * self.mean_perc / 100
            # note that here variance has units of nA, not nA^2
            self.var = threshold * self.var_perc / 100

    def draw_already_used_numbers(self, rng, dt):
        prev_t = 0
        tstep = self.duration - dt

        while prev_t < self.delay - dt:
            next_t = min(prev_t + tstep, self.delay - dt)

            tvec = Nd.h.Vector()
            tvec.indgen(prev_t, next_t, self.dt)
            stim = Nd.h.Vector(len(tvec))
            stim.setrand(rng)

            prev_t = next_t + dt


@StimulusManager.register_type
class Pulse(BaseStim):
    """Inject a pulse train with given amplitude, frequency and width."""

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

        self.stimList = []  # CurrentSource's go here

        if not self.parse_check_all_parameters(stim_info):
            return  # nothing to do, stim is a no-op

        # apply stim to each point in target
        tpoints = target.getPointList(cell_manager)
        for tpoint_list in tpoints:
            for sec_id, sc in enumerate(tpoint_list.sclst):
                # skip sections not in this split
                if not sc.exists():
                    continue

                # generate pulse train current source
                cs = CurrentSource.train(
                    self.amp,
                    self.freq,
                    self.width,
                    self.duration,
                    delay=self.delay,
                    represents_physical_electrode=self.represents_physical_electrode,
                )
                # attach current source to section
                cs.attach_to(sc.sec, tpoint_list.x[sec_id])
                self.stimList.append(cs)  # save CurrentSource

    def parse_check_all_parameters(self, stim_info: dict):
        self.amp = float(stim_info["AmpStart"])  # amplitude [nA]
        self.freq = float(stim_info["Frequency"])  # frequency [Hz]
        self.width = float(stim_info["Width"])  # pulse width [ms]

        return self.freq > 0 and self.width > 0  # no-op if any is 0


@StimulusManager.register_type
class Sinusoidal(BaseStim):
    """Inject a sinusoidal current with given amplitude and frequency."""

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

        self.stimList = []  # CurrentSource's go here

        if not self.parse_check_all_parameters(stim_info):
            return  # nothing to do, stim is a no-op

        # apply stim to each point in target
        tpoints = target.getPointList(cell_manager)
        for tpoint_list in tpoints:
            for sec_id, sc in enumerate(tpoint_list.sclst):
                # skip sections not in this split
                if not sc.exists():
                    continue

                # generate sinusoidal current source
                cs = CurrentSource.sin(
                    self.amp,
                    self.duration,
                    self.freq,
                    step=self.dt,
                    delay=self.delay,
                    represents_physical_electrode=self.represents_physical_electrode,
                )
                # attach current source to section
                cs.attach_to(sc.sec, tpoint_list.x[sec_id])
                self.stimList.append(cs)  # save CurrentSource

    def parse_check_all_parameters(self, stim_info: dict):
        self.dt = float(stim_info.get("Dt", 0.025))  # stimulus timestep [ms]
        if self.dt <= 0:
            raise Exception(f"{self.__class__.__name__} time-step must be positive")

        self.amp = float(stim_info["AmpStart"])  # amplitude [nA]
        self.freq = float(stim_info["Frequency"])  # frequency [Hz]

        return self.freq > 0  # no-op if 0


@StimulusManager.register_type
class SEClamp(BaseStim):
    """Apply a single electrode voltage clamp."""

    def __init__(self, target, stim_info: dict, cell_manager):
        super().__init__(target, stim_info, cell_manager)

        self.stimList = []  # SEClamp's go here

        self.parse_check_all_parameters(stim_info)

        # apply stim to each point in target
        tpoints = target.getPointList(cell_manager)
        for tpoint_list in tpoints:
            for sec_id, sc in enumerate(tpoint_list.sclst):
                # skip sections not in this split
                if not sc.exists():
                    continue

                # create single electrode voltage clamp at location
                seclamp = Nd.h.SEClamp(tpoint_list.x[sec_id], sec=sc.sec)

                seclamp.rs = self.rs
                seclamp.dur1 = self.duration
                seclamp.amp1 = self.vhold
                self.stimList.append(seclamp)  # save SEClamp

    def parse_check_all_parameters(self, stim_info: dict):
        self.vhold = float(stim_info["Voltage"])  # holding voltage [mV]
        self.rs = float(stim_info.get("RS", 0.01))  # series resistance [MOhm]
        if self.delay > 0:
            logging.warning("%s ignores delay", self.__class__.__name__)
