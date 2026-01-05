"""Runtime configuration"""

import logging
import os
import os.path
import re
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ._shmutils import SHMUtil
from neurodamus.io.sonata_config import SonataConfig
from neurodamus.utils.logging import log_verbose
from neurodamus.utils.pyutils import ConfigT, StrEnumBase

EXCEPTION_NODE_FILENAME = ".exception_node"
"""A file which controls which rank shows exception"""


class LogLevel:
    ERROR_ONLY = 0
    DEFAULT = 1
    VERBOSE = 2
    DEBUG = 3


class ConfigurationError(Exception):
    """Error due to invalid settings in simulation config
    ConfigurationError should be raised by all ranks to be caught
    properly. Otherwise we might end up with deadlocks.
    For Exceptions that are raised by a single rank Exception
    should be used.
    This is due to the way Exceptions are caught from commands.py.
    """


class GlobalConfig:
    verbosity = LogLevel.DEFAULT
    debug_conn = os.getenv("ND_DEBUG_CONN", "")
    if debug_conn:
        debug_conn = [int(gid) for gid in debug_conn.split(",")]
        verbosity = 3

    @classmethod
    def set_mpi(cls):
        os.environ["NEURON_INIT_MPI"] = "1"


class Feature(Enum):
    Replay = 1
    SpontMinis = 2
    SynConfigure = 3
    Stimulus = 4
    LoadBalance = 5


class CellPermute(StrEnumBase):
    UNPERMUTED = 0  # cpu
    NODE_ADJACENCY = 1  # cpu/gpu

    __mapping__ = [
        ("unpermuted", UNPERMUTED),
        ("node-adjacency", NODE_ADJACENCY),
    ]
    __default__ = UNPERMUTED


class CliOptions(ConfigT):
    cell_permute = None
    report_buffer_size = None
    build_model = None
    simulate_model = True
    output_path = None
    keep_build = False
    lb_mode = None
    modelbuilding_steps = None
    enable_coord_mapping = False
    save = False
    restore = None
    enable_shm = False
    model_stats = False
    simulator = None
    dry_run = False
    num_target_ranks = None
    keep_axon = False
    coreneuron_direct_mode = False
    crash_test = False

    # Restricted Functionality support, mostly for testing

    class NoRestriction:
        """Provide container API, where `in` checks are always True"""

        def __contains__(self, _) -> bool:
            return True

    NoRestriction = NoRestriction()  # Singleton

    restrict_features = NoRestriction  # can also be a list
    restrict_node_populations = NoRestriction
    restrict_connectivity = 0  # no restriction, 1 to disable projections, 2 to disable all


class CircuitConfig(ConfigT):
    name = None
    Engine = None
    nrnPath = ConfigT.REQUIRED  # noqa: N815
    CellLibraryFile = ConfigT.REQUIRED
    METypePath = None
    MorphologyType = None
    MorphologyPath = None
    PopulationName = None
    NodesetName = None
    DetailedAxon = False
    PopulationType = None


class LoadBalanceMode(Enum):
    """An enumeration, inc parser, of the load balance modes."""

    RoundRobin = 0
    WholeCell = 1
    MultiSplit = 2
    Memory = 3

    @classmethod
    def parse(cls, lb_mode):
        """Parses the load balancing mode from a string.
        Options other than WholeCell or MultiSplit are considered RR
        """
        if lb_mode is None:
            return None
        modes = {
            "rr": cls.RoundRobin,
            "roundrobin": cls.RoundRobin,
            "wholecell": cls.WholeCell,
            "loadbalance": cls.MultiSplit,
            "multisplit": cls.MultiSplit,
            "memory": cls.Memory,
        }
        lb_mode_enum = modes.get(lb_mode.lower())
        if lb_mode_enum is None:
            raise ConfigurationError("Unknown load balance mode: " + lb_mode)
        return lb_mode_enum

    class AutoBalanceModeParams:
        """Parameters for auto-selecting a load-balance mode"""

        multisplit_cpu_cell_ratio = 4  # For warning
        cell_count = 1000
        duration = 1000
        mpi_ranks = 200

    @classmethod
    def auto_select(cls, use_neuron, cell_count, duration, auto_params=AutoBalanceModeParams):
        """Simple heuristics for auto selecting load balance"""
        from ._neurodamus import MPI

        lb_mode = LoadBalanceMode.RoundRobin
        reason = ""
        if MPI.size == 1:
            lb_mode = LoadBalanceMode.RoundRobin
            reason = "Single rank - not worth using Load Balance"
        elif use_neuron and MPI.size >= auto_params.multisplit_cpu_cell_ratio * cell_count:
            logging.warning(
                "There's potentially a high number of empty ranks. "
                "To activate multi-split set --lb-mode=MultiSplit"
            )
        elif (
            cell_count > auto_params.cell_count
            and duration > auto_params.duration
            and MPI.size > auto_params.mpi_ranks
        ):
            lb_mode = LoadBalanceMode.WholeCell
            reason = "Simulation size"
        return lb_mode, reason


class _SimConfig:
    """A class initializing several HOC config objects and proxying to simConfig"""

    config_file = None
    cli_options = None
    run_conf = None
    output_root = None
    sonata_circuits = None
    projections = None
    connections = None
    stimuli = None
    reports = None
    modifications = None
    beta_features = None

    # Hoc objects used
    _config_parser = None
    _parsed_run = None
    _simulation_config = None
    _simconf = None
    rng_info = None

    # In principle not all vars need to be required as they'r set by the parameter functions
    simulation_config_dir = None
    default_neuron_dt = 0.025
    buffer_time = 25
    save = None
    restore = None
    coreneuron_datadir = None
    extracellular_calcium = None
    use_coreneuron = False
    use_neuron = True
    report_buffer_size = 8  # in MB
    cell_permute = CellPermute.default()
    delete_corenrn_data = False
    modelbuilding_steps = 1
    build_model = True
    simulate_model = True
    loadbal_mode = None
    spike_location = "soma"
    spike_threshold = -30
    dry_run = False
    num_target_ranks = None
    coreneuron_direct_mode = False
    crash_test_mode = False

    _validators = []
    _cell_requirements = {}

    restore_coreneuron = property(lambda self: self.use_coreneuron and bool(self.restore))
    cell_requirements = property(lambda self: self._cell_requirements)

    @classmethod
    def init(cls, config_file, cli_options):
        # Import these objects scope-level to avoid cross module dependency
        from . import NeuronWrapper as Nd

        Nd.init()
        if not os.path.isfile(config_file):
            raise ConfigurationError("Config file not found: " + config_file)
        logging.info("Initializing Simulation Configuration and Validation")

        log_verbose("ConfigFile: %s", config_file)
        log_verbose("CLI Options: %s", cli_options)
        cls.config_file = config_file
        cls._config_parser = cls._init_config_parser(config_file)
        cls._parsed_run = cls._config_parser.parsedRun
        cls._simulation_config = cls._config_parser  # Please refactor me
        cls.simulation_config_dir = os.path.dirname(os.path.abspath(config_file))
        log_verbose(
            "SimulationConfigDir using directory of simulation config file: %s",
            cls.simulation_config_dir,
        )

        cls.projections = cls._config_parser.parsedProjections
        cls.connections = cls._config_parser.parsedConnects
        cls.stimuli = cls._config_parser.parsedStimuli
        cls.reports = cls._config_parser.parsedReports
        cls.modifications = cls._config_parser.parsedModifications or {}
        cls.beta_features = cls._config_parser.beta_features
        cls.cli_options = CliOptions(**(cli_options or {}))

        cls.dry_run = cls.cli_options.dry_run
        cls.crash_test_mode = cls.cli_options.crash_test
        cls.num_target_ranks = cls.cli_options.num_target_ranks
        # change simulator by request before validator and init hoc config
        if cls.cli_options.simulator:
            cls._parsed_run["Simulator"] = cls.cli_options.simulator

        cls.run_conf = cls._parsed_run
        for validator in cls._validators:
            validator(cls)

        logging.info("Initializing hoc config objects")
        cls._init_hoc_config_objs()

    @classmethod
    def build_path(cls):
        """Default to <currend_dir>/build if save is None"""
        return cls.save or str(Path(cls.simulation_config_dir) / "build")

    @classmethod
    def coreneuron_datadir_path(cls, create=False):
        """Default to output_root if none is provided"""
        ans = cls.coreneuron_datadir or str(Path(cls.build_path()) / "coreneuron_input")
        if create:
            Path(ans).mkdir(parents=True, exist_ok=True)
        return ans

    @classmethod
    def coreneuron_datadir_restore_path(cls):
        """Default to output_root if none is provided"""
        return str(Path(cls.restore) / "coreneuron_input")

    @classmethod
    def populations_offset_restore_path(cls):
        return str(Path(cls.restore) / "populations_offset.dat")

    @classmethod
    def populations_offset_save_path(cls, create=False):
        """Get polulations_offset path for the save folder.

        Used internally for a save/restore cycle.

        Optional: create pathing folders if necessary
        """
        ans = Path(cls.build_path()) / "populations_offset.dat"
        if create:
            ans.parent.mkdir(parents=True, exist_ok=True)
        return str(ans)

    @classmethod
    def populations_offset_output_path(cls, create=False):
        """Get polulations_offset path for the output folder.

        Used for visualization.

        Optional: create pathing folders if necessary
        """
        ans = Path(cls.output_root) / "populations_offset.dat"
        if create:
            ans.parent.mkdir(parents=True, exist_ok=True)
        return str(ans)

    @classmethod
    def output_root_path(cls, create=False):
        """Get the output_root path

        Create the folder path if required and needed
        """
        outdir = Path(SimConfig.output_root)
        if create:
            outdir.mkdir(parents=True, exist_ok=True)
        return str(outdir)

    @classmethod
    def _init_config_parser(cls, config_file):
        if not config_file.endswith(".json"):
            raise ConfigurationError(
                "Invalid configuration file format. The configuration file must be a .json file."
            )
        try:
            config_parser = SonataConfig(config_file)
        except Exception as e:
            msg = f"Failed to initialize SonataConfig with {config_file}"
            raise ConfigurationError(msg) from e
        return config_parser

    @classmethod
    def _init_hoc_config_objs(cls):
        """Init objects which parse/check configs in the hoc world"""
        from neuron import h

        parsed_run = cls._parsed_run

        cls.rng_info = h.RNGSettings()
        cls.rng_info.interpret(parsed_run, cls.use_coreneuron)
        if "BaseSeed" in parsed_run:
            logging.info("User-defined RNG base seed %s", parsed_run["BaseSeed"])

    @classmethod
    def validator(cls, f):
        """Decorator to register parameters / config validators"""
        cls._validators.append(f)

    @classmethod
    def update_connection_blocks(cls, alias):
        """Convert source destination to real population names

        Args:
            alias: A dict associating alias->real_name's
        """
        from neurodamus.target_manager import TargetSpec  # avoid cyclic deps

        restrict_features = SimConfig.cli_options.restrict_features
        if Feature.SpontMinis not in restrict_features:
            logging.warning("Disabling SpontMinis (restrict_features)")
        if Feature.SynConfigure not in restrict_features:
            logging.warning("Disabling SynConfigure (restrict_features)")

        def update_item(conn, item):
            src_spec = TargetSpec(conn.get(item), None)
            src_spec.population = alias.get(src_spec.population, src_spec.population)
            conn[item] = str(src_spec)

        new_connections = {}
        for name, conn in cls.connections.items():
            update_item(conn, "Source")
            update_item(conn, "Destination")
            if Feature.SpontMinis not in restrict_features:
                conn.pop("SpontMinis", None)
            if Feature.SynConfigure not in restrict_features:
                conn.pop("SynapseConfigure", None)

            new_connections[name] = conn

        cls.connections = new_connections

    @classmethod
    def check_connections_configure(cls, target_manager):
        check_connections_configure(cls, target_manager)  # top_level

    @classmethod
    def check_cell_requirements(cls, target_manager):
        _input_resistance(cls, target_manager)  # top_level


# Singleton
SimConfig = _SimConfig()


def _check_params(  # noqa: C901
    section_name,
    data,
    required_fields,
    numeric_fields=(),
    non_negatives=(),
    valid_values=None,
    deprecated_values=None,
):
    """Generic function to check a dict-like data set conforms to the field prescription"""
    for param in required_fields:
        if param not in data:
            raise ConfigurationError(
                f"simulation config mandatory param not present: [{section_name}] {param}"
            )
    for param in set(numeric_fields + non_negatives):
        val = data.get(param)
        try:
            val and float(val)
        except ValueError as e:
            raise ConfigurationError(
                f"simulation config param must be numeric: [{section_name}] {param}"
            ) from e
    for param in non_negatives:
        val = data.get(param)
        if val and float(val) < 0:
            raise ConfigurationError(
                f"simulation config param must be positive: [{section_name}] {param}"
            )

    for param, valid in (valid_values or {}).items():
        val = data.get(param)
        if val and val not in valid:
            raise ConfigurationError(
                f"simulation config param value is invalid: [{section_name}] {param} = {val}"
            )

    for param, deprecated in (deprecated_values or {}).items():
        val = data.get(param)
        if val and val in deprecated:
            logging.warning(
                "simulation config param value is deprecated: [%s] %s = %s",
                section_name,
                param,
                val,
            )


@SimConfig.validator
def _run_params(config: _SimConfig):
    required_fields = ("Duration",)
    numeric_fields = ("BaseSeed", "StimulusSeed", "Celsius", "V_Init")
    non_negatives = ("Duration", "Dt", "ModelBuildingSteps")
    _check_params("Run default", config.run_conf, required_fields, numeric_fields, non_negatives)


@SimConfig.validator
def _loadbal_mode(config: _SimConfig):
    cli_args = config.cli_options
    if Feature.LoadBalance not in cli_args.restrict_features:
        logging.warning("Disabled Load Balance (restrict_features)")
        config.loadbal_mode = LoadBalanceMode.RoundRobin
        return
    lb_mode_str = cli_args.lb_mode or config.run_conf.get("RunMode")
    config.loadbal_mode = LoadBalanceMode.parse(lb_mode_str)


@SimConfig.validator
def _projection_params(config: _SimConfig):
    required_fields = ("Path",)
    for name, proj in config.projections.items():
        _check_params("Projection " + name, proj, required_fields, (), ())
        _validate_file_extension(proj.get("Path"))


@SimConfig.validator
def _stimulus_params(config: _SimConfig):
    required_fields = (
        "Mode",
        "Pattern",
        "Duration",
        "Delay",
    )
    numeric_fields = (
        "Dt",
        "RiseTime",
        "DecayTime",
        "AmpMean",
        "AmpVar",
        "MeanPercent",
        "SDPercent",
        "RelativeSkew",
        "AmpStart",
        "AmpEnd",
        "PercentStart",
        "PercentEnd",
        "PercentLess",
        "Mean",
        "Variance",
        "Voltage",
        "RS",
    )
    non_negatives = (
        "Duration",
        "Delay",
        "Rate",
        "Frequency",
        "Width",
        "Lambda",
        "Weight",
        "NumOfSynapses",
        "Seed",
    )
    valid_values = {
        "Mode": ("Current", "Voltage", "Conductance", "spikes"),
        "Pattern": {
            "Hyperpolarizing",
            "Linear",
            "Noise",
            "Pulse",
            "RelativeLinear",
            "RelativeShotNoise",
            "SEClamp",
            "ShotNoise",
            "Sinusoidal",
            "SubThreshold",
            "SynapseReplay",
            "OrnsteinUhlenbeck",
            "NPoisson",
            "NPoissonInhomogeneous",
            "ReplayVoltageTrace",
            "AbsoluteShotNoise",
            "RelativeOrnsteinUhlenbeck",
        },
    }
    deprecated_values = {"Pattern": ("NPoisson", "NPoissonInhomogeneous", "ReplayVoltageTrace")}
    for stim in config.stimuli:
        _check_params(
            "Stimulus " + stim["Name"],
            stim,
            required_fields,
            numeric_fields,
            non_negatives,
            valid_values,
            deprecated_values,
        )


@SimConfig.validator
def _modification_params(config: _SimConfig):
    required_fields = (
        "Target",
        "Type",
    )
    for name, mod_block in config.modifications.items():
        _check_params("Modification " + name, mod_block, required_fields, ())


def make_circuit_config(config_dict, req_morphology=True):
    if config_dict.get("CellLibraryFile") == "<NONE>":
        config_dict["CellLibraryFile"] = False
        config_dict["nrnPath"] = False
        config_dict["MorphologyPath"] = False
    elif config_dict.get("nrnPath") == "<NONE>":
        config_dict["nrnPath"] = False
    _validate_circuit_morphology(config_dict, req_morphology)
    _validate_file_extension(config_dict.get("CellLibraryFile"))
    _validate_file_extension(config_dict.get("nrnPath"))
    return CircuitConfig(config_dict)


def _validate_circuit_morphology(config_dict, required=True):
    morph_path = config_dict.get("MorphologyPath")
    morph_type = config_dict.get("MorphologyType")
    if morph_path is None and required:
        raise ConfigurationError("No morphology path provided (Required!)")
    # Some circuit types may not require morphology files
    if not morph_path:
        log_verbose(" > Morphology src: <Disabled> MorphologyType: %s, ", morph_type or "<None>")
        return
    if morph_type is None:
        logging.warning("MorphologyType not set. Assuming ascii and legacy /ascii subdir")
        morph_type = "asc"
        morph_path = os.path.join(morph_path, "ascii")
        config_dict["MorphologyType"] = morph_type
        config_dict["MorphologyPath"] = morph_path
    assert morph_type in {"asc", "swc", "h5", "hoc"}, "Invalid MorphologyType"
    log_verbose(" > MorphologyType = %s, src: %s", morph_type, morph_path)


def _validate_file_extension(path):
    if not path:
        return
    filepath = path.split(":")[0]  # for sonata, remove the edge_pop name appended to the file path
    if filepath.endswith(".sonata"):
        raise ConfigurationError(
            "*.sonata files are no longer supported, please rename them to *.h5"
        )


@SimConfig.validator
def _circuits(config: _SimConfig):
    from . import EngineBase

    circuit_configs = {}

    for name, circuit_info in config._simulation_config.Circuit.items():
        log_verbose("CIRCUIT %s (%s)", name, circuit_info.get("Engine", "(default)"))
        # Replace name by actual engine class
        circuit_info["Engine"] = EngineBase.get(circuit_info["Engine"])

        circuit_info.setdefault("nrnPath", False)
        if config.cli_options.keep_axon and circuit_info["Engine"].__name__ == "METypeEngine":
            log_verbose("Keeping axons ENABLED")
            circuit_info.setdefault("DetailedAxon", True)

        circuit_configs[name] = make_circuit_config(circuit_info, req_morphology=False)
        circuit_configs[name].name = name

    # Sort so that iteration is deterministic
    config.sonata_circuits = dict(
        sorted(
            circuit_configs.items(),
            key=lambda x: (x[1].Engine.CircuitPrecedence if x[1].Engine else 0, x[0]),
        )
    )


@SimConfig.validator
def _global_parameters(config: _SimConfig):
    from neuron import h

    run_conf = config.run_conf

    config.celsius = run_conf.get("Celsius", 34)
    config.v_init = run_conf.get("V_Init", -65)
    config.extracellular_calcium = run_conf.get("ExtracellularCalcium")
    config.buffer_time = 25 * run_conf.get("FlushBufferScalar", 1)
    config.tstop = run_conf["Duration"]
    h.celsius = config.celsius
    h.set_v_init(config.v_init)
    h.tstop = config.tstop
    config.default_neuron_dt = h.dt
    h.dt = run_conf.get("Dt", h.dt)
    h.steps_per_ms = 1.0 / h.dt
    props = ("celsius", "v_init", "extracellular_calcium", "tstop", "buffer_time")
    log_verbose("Global params: %s", " | ".join(p + f": {getattr(config, p)}" for p in props))
    if "CompartmentsPerSection" in run_conf:
        logging.warning(
            "CompartmentsPerSection is currently not supported. "
            "If needed, please request with a detailed usecase."
        )


@SimConfig.validator
def _set_simulator(config: _SimConfig):
    user_config = config.cli_options
    simulator = config.run_conf.get("Simulator")

    if simulator is None:
        config.run_conf["Simulator"] = simulator = "NEURON"
    if simulator not in {"NEURON", "CORENEURON"}:
        raise ConfigurationError("'Simulator' value must be either NEURON or CORENEURON")
    if simulator == "NEURON" and (
        user_config.build_model is False or user_config.simulate_model is False
    ):
        raise ConfigurationError(
            "Disabling model building or simulation is only compatible with CoreNEURON"
        )

    log_verbose("Simulator = %s", simulator)
    config.use_neuron = simulator == "NEURON"
    config.use_coreneuron = simulator == "CORENEURON"


@SimConfig.validator
def _spike_parameters(config: _SimConfig):
    spike_location = config.run_conf.get("SpikeLocation", "soma")
    if spike_location not in {"soma", "AIS"}:
        raise ConfigurationError("Possible options for SpikeLocation are 'soma' and 'AIS'")
    spike_threshold = config.run_conf.get("SpikeThreshold", -30)
    log_verbose("Spike_Location = %s", spike_location)
    log_verbose("Spike_Threshold = %s", spike_threshold)
    config.spike_location = spike_location
    config.spike_threshold = spike_threshold


_condition_checks = {
    "secondorder": (
        (0, 1, 2),
        ConfigurationError(
            "Time integration method (SecondOrder value) {} is invalid. Valid options are:"
            " '0' (implicitly backward euler),"
            " '1' (Crank-Nicolson) and"
            " '2' (Crank-Nicolson with fixed ion currents)"
        ),
    ),
    "randomize_Gaba_risetime": (
        ("True", "False", "0", "false"),
        ConfigurationError("randomize_Gaba_risetime must be True or False"),
    ),
}


@SimConfig.validator
def _simulator_globals(config: _SimConfig):
    from neuron import h

    # Hackish but some constants only live in the helper
    h.load_file("GABAABHelper.hoc")

    for group in config._simulation_config.Conditions.values():
        for key, value in group.items():
            validator = _condition_checks.get(key)
            if validator:
                config_exception = validator[1] or ConfigurationError(
                    f"Value {value} not valid for key {key}. Allowed: {validator[0]}"
                )
                if value not in validator[0]:
                    raise config_exception

            log_verbose("GLOBAL %s = %s", key, value)

            setattr(h, key, value)

            if "cao_CR" in key and value != config.extracellular_calcium:
                logging.warning(
                    "Value of %s (%s) is not the same as extracellular_calcium (%s)",
                    key,
                    value,
                    config.extracellular_calcium,
                )


@SimConfig.validator
def _second_order(config: _SimConfig):
    second_order = config.run_conf.get("SecondOrder")
    if second_order is None:
        return
    second_order = int(second_order)
    if second_order in {0, 1, 2}:
        from neuron import h

        log_verbose("SecondOrder = %g", second_order)
        config.second_order = second_order
        h.secondorder = second_order
    else:
        raise _condition_checks["secondorder"][1]


@SimConfig.validator
def _single_vesicle(config: _SimConfig):
    if "MinisSingleVesicle" not in config.run_conf:
        return

    from neuron import h

    if not hasattr(h, "minis_single_vesicle_ProbAMPANMDA_EMS"):
        raise NotImplementedError(
            "Synapses don't implement minis_single_vesicle. More recent neurodamus model required."
        )
    minis_single_vesicle = int(config.run_conf["MinisSingleVesicle"])
    log_verbose("minis_single_vesicle = %d", minis_single_vesicle)
    h.minis_single_vesicle_ProbAMPANMDA_EMS = minis_single_vesicle
    h.minis_single_vesicle_ProbGABAAB_EMS = minis_single_vesicle
    h.minis_single_vesicle_GluSynapse = minis_single_vesicle


@SimConfig.validator
def _randomize_gaba_risetime(config: _SimConfig):
    randomize_risetime = config.run_conf.get("RandomizeGabaRiseTime")
    if randomize_risetime is None:
        return
    from neuron import h

    h.load_file("GABAABHelper.hoc")
    if not hasattr(h, "randomize_Gaba_risetime"):
        raise NotImplementedError(
            "Models don't support setting RandomizeGabaRiseTime. "
            "Please load a more recent model or drop the option."
        )
    assert randomize_risetime in {"True", "False", "0", "false"}  # any non-"True" value is negative
    log_verbose("randomize_Gaba_risetime = %s", randomize_risetime)
    h.randomize_Gaba_risetime = randomize_risetime


@SimConfig.validator
def _output_root(config: _SimConfig):
    """Confirm output_path exists and is usable"""
    output_path = config.run_conf.get("OutputRoot")

    if config.cli_options.output_path not in {None, output_path}:
        output_path = config.cli_options.output_path
    if output_path is None:
        raise ConfigurationError("'OutputRoot' configuration not set")
    if not Path(output_path).is_absolute():
        output_path = Path(config.simulation_config_dir) / output_path

    log_verbose("OutputRoot = %s", output_path)
    config.run_conf["OutputRoot"] = str(output_path)
    config.output_root = str(output_path)


@SimConfig.validator
def _check_save(config: _SimConfig):
    cli_args = config.cli_options
    save_path = cli_args.save or config.run_conf.get("Save")

    if not save_path:
        return

    if not config.use_coreneuron:
        raise ConfigurationError("Save-restore only available with CoreNeuron")

    # Handle save
    assert isinstance(save_path, str), "Save must be a string path"
    path_obj = Path(save_path)
    if not path_obj.is_absolute():
        path_obj = Path(config.simulation_config_dir) / path_obj

    config.save = str(path_obj)


@SimConfig.validator
def _check_restore(config: _SimConfig):
    restore = config.cli_options.restore or config.run_conf.get("Restore")
    if not restore:
        return

    if not config.use_coreneuron:
        raise ConfigurationError("Save-restore only available with CoreNeuron")

    # sync restore settings to hoc, otherwise we end up with an empty coreneuron_input dir
    config.run_conf["Restore"] = restore

    assert isinstance(restore, str), "Restore must be a string path"
    path_obj = Path(restore)
    if not path_obj.is_absolute():
        path_obj = Path(config.simulation_config_dir) / path_obj

    config.restore = str(path_obj)


@SimConfig.validator
def _check_model_build_mode(config: _SimConfig):
    user_config = config.cli_options
    config.build_model = user_config.build_model
    config.simulate_model = user_config.simulate_model

    if config.build_model is True:  # just create the data
        return
    if config.use_coreneuron is False:
        if config.build_model is False:
            raise ConfigurationError("Skipping model build is only available with CoreNeuron")
        # was None
        config.build_model = True
        return

    # CoreNeuron restore is a bit special and already had its input dir checked
    if config.restore:
        config.build_model = False
        return

    # It's a CoreNeuron run. We have to check if build_model is AUTO or OFF
    core_data_location = config.coreneuron_datadir_path()

    try:
        # Ensure that 'sim.conf' and 'files.dat' exist, and that '/dev/shm' was not used
        contents = (Path(core_data_location).parent / "sim.conf").read_text()
        core_data_exists = (
            "datpath='/dev/shm/" not in contents and Path(core_data_location, "files.dat").is_file()
        )
    except FileNotFoundError:
        core_data_exists = False

    if config.build_model in {None, "AUTO"}:
        # If enable-shm option is given we have to rebuild the model and delete any previous files
        # in /dev/shm or gpfs
        # In any case when we start model building any data in the core_data_location_shm if it
        # exists are deleted
        if not core_data_exists:
            core_data_location_shm = SHMUtil.get_datadir_shm(core_data_location)
            data_location = (
                core_data_location_shm
                if (user_config.enable_shm and core_data_location_shm is not None)
                else core_data_location
            )
            logging.info(
                "Valid CoreNeuron input data not found in '%s'. "
                "Neurodamus will proceed to model building.",
                data_location,
            )
            config.build_model = True
        else:
            logging.info(
                "CoreNeuron input data found in %s. Skipping model build", core_data_location
            )
            config.build_model = False

    if not config.build_model and not core_data_exists:
        raise ConfigurationError("Model build DISABLED but no CoreNeuron data found")


@SimConfig.validator
def _keep_coreneuron_data(config: _SimConfig):
    if config.use_coreneuron:
        keep_core_data = False
        if config.cli_options.keep_build or config.run_conf.get("KeepModelData", False) == "True":
            keep_core_data = True
        elif not config.cli_options.simulate_model or config.save:
            logging.warning("Keeping coreneuron data for CoreNeuron following run")
            keep_core_data = True
        config.delete_corenrn_data = not keep_core_data
    log_verbose("delete_corenrn_data = %s", config.delete_corenrn_data)


@SimConfig.validator
def _report_buffer_size(config: _SimConfig):
    user_config = config.cli_options
    if user_config.report_buffer_size is not None:
        report_buffer_size = int(user_config.report_buffer_size)
    else:
        return

    if config.restore is not None:
        raise ConfigurationError("--report-buffer-size cannot be used with --restore")

    assert report_buffer_size > 0, "Report buffer size must be > 0"
    config.report_buffer_size = report_buffer_size


@SimConfig.validator
def _model_building_steps(config: _SimConfig):
    user_config = config.cli_options
    if user_config.modelbuilding_steps is not None:
        ncycles = int(user_config.modelbuilding_steps)
    else:
        return

    assert ncycles > 0, "ModelBuildingSteps set to 0. Required value > 0"

    if not SimConfig.use_coreneuron:
        logging.warning("IGNORING ModelBuildingSteps since simulator is not CORENEURON")
        return

    if "NodesetName" not in config.run_conf:
        raise ConfigurationError("Multi-iteration coreneuron data generation requires NodesetName")

    logging.info("Splitting Target for multi-iteration CoreNeuron data generation")
    logging.info(" -> Cycles: %d. [src: %s]", ncycles, "CLI")
    config.modelbuilding_steps = ncycles


@SimConfig.validator
def _spikes_sort_order(config: _SimConfig):
    order = config.run_conf.get("SpikesSortOrder", "by_time")
    if order not in {"none", "by_time"}:
        raise ConfigurationError(
            f"Unsupported spikes sort order {order}, " + "BBP supports 'none' and 'by_time'"
        )


@SimConfig.validator
def _coreneuron_direct_mode(config: _SimConfig):
    user_config = config.cli_options
    direct_mode = user_config.coreneuron_direct_mode
    if direct_mode:
        if config.use_neuron:
            raise ConfigurationError("--coreneuron-direct-mode is not valid for NEURON")
        if config.modelbuilding_steps > 1:
            logging.warning(
                "--coreneuron-direct-mode not valid for multi-cyle model building, "
                "continue with file mode"
            )
            direct_mode = False
        if config.save or config.restore:
            logging.warning(
                "--coreneuron-direct-mode not valid for save/restore, continue with file mode"
            )
            direct_mode = False

    if direct_mode:
        logging.info("Run CORENEURON direct mode without writing model data to disk")
    config.coreneuron_direct_mode = direct_mode


@SimConfig.validator
def _cell_permute(config: _SimConfig):
    user_config = config.cli_options
    if user_config.cell_permute is not None:
        config.cell_permute = CellPermute.from_string(str(user_config.cell_permute))
    if config.use_neuron and config.cell_permute != CellPermute.UNPERMUTED:
        raise ConfigurationError(
            f"Cell permutation is only available with CoreNEURON. "
            f"--cell-permute={config.cell_permute.to_string()} is invalid with NEURON."
        )


def get_debug_cell_gids(cli_options):
    """Parse the --dump-cell-state option from CLI.

    Supports:
    - A single integer (e.g., "2")
    - A comma-separated list of integers or ranges (e.g., "1,3-5,7")
    - A file path containing one GID per line

    Returns:
        List of GIDs as integers, or None if not provided.

    Raises:
        ConfigurationError: if the format is invalid or file doesn't exist.
    """
    value = cli_options.get("dump_cell_state") if cli_options else None
    if value is None:
        return None

    def parse_gid_token(token):
        """Parse a single token: an integer or a range (e.g., "3" or "1-4").

        Returns:
            List of integers parsed from the token.

        Raises:
            ConfigurationError: if the token is invalid.
        """
        token = token.strip()
        if re.match(r"^\d+-\d+$", token):
            start, end = map(int, token.split("-"))
            if start > end:
                raise ConfigurationError(f"Invalid range in dump-cell-state: {token}")
            return list(range(start, end + 1))
        if token.isdigit():
            return [int(token)]
        raise ConfigurationError(f"Invalid token in dump-cell-state: {token}")

    try:
        tokens = [str(value)] if isinstance(value, int) else value.split(",")
        gids = []
        for token in tokens:
            gids.extend(parse_gid_token(token))
        gids = list(dict.fromkeys(gids))  # Remove duplicates while preserving order
    except ValueError as e:
        raise ConfigurationError("Cannot parse dump-cell-state: " + value) from e
    else:
        return gids


@dataclass
class _ConnCtx:
    conf: dict[str, str]
    override: "_ConnCtx | None" = None
    full_override: bool = False
    visited: bool = False


def _get_syn_config_vars(conn_ctx: _ConnCtx, config_assignment: re.Pattern[str]) -> list[str]:
    """Return the list of synapse configuration variables from SynapseConfigure field."""
    return [var for var, _ in config_assignment.findall(conn_ctx.conf.get("SynapseConfigure", ""))]


def _get_overlapping_connection_pathway(
    base_conns: list[_ConnCtx], conn_ctx: _ConnCtx, target_manager
) -> Generator[_ConnCtx, None, None]:
    """Yield previously processed connections that overlap with conn_ctx."""
    for base_conn in reversed(base_conns):
        if target_manager.pathways_overlap(base_conn.conf, conn_ctx.conf):
            yield base_conn


def _build_override_chains(
    all_conn_blocks: list[_ConnCtx],
    processed_conn_blocks: list[_ConnCtx],
    zero_weight_conns: list[_ConnCtx],
    conn_configure_global_vars: dict[str, list[str]],
    config_assignment: re.Pattern[str],
    target_manager,
) -> None:
    """Phase 1: Process t=0 connections, build override chains and collect global variables."""
    for conn_ctx in all_conn_blocks:
        if float(conn_ctx.conf.get("Delay", 0)) > 0:
            continue

        for var in _get_syn_config_vars(conn_ctx, config_assignment):
            if not var.startswith("%s"):
                conn_configure_global_vars[conn_ctx.conf["Name"]].append(var)

        if float(conn_ctx.conf.get("Weight", 1)) == 0:
            zero_weight_conns.append(conn_ctx)

        for overridden_conn in _get_overlapping_connection_pathway(
            processed_conn_blocks, conn_ctx, target_manager
        ):
            conn_ctx.override = overridden_conn
            if target_manager.pathways_overlap(
                conn_ctx.conf, overridden_conn.conf, equal_only=True
            ):
                overridden_conn.full_override = True
            break

        processed_conn_blocks.append(conn_ctx)


def _process_delayed_connections(
    all_conn_blocks: list[_ConnCtx], zero_weight_conns: list[_ConnCtx], target_manager
) -> None:
    """Phase 2: Process delayed connections (Delay>0) overriding t=0 weight=0 blocks."""
    for conn_ctx in all_conn_blocks:
        if float(conn_ctx.conf.get("Delay", 0)) == 0:
            continue
        is_overriding = False
        for base_conn in _get_overlapping_connection_pathway(
            zero_weight_conns, conn_ctx, target_manager
        ):
            is_overriding = True
            if not base_conn.full_override:
                base_conn.full_override = target_manager.pathways_overlap(
                    conn_ctx.conf, base_conn.conf, equal_only=True
                )
        if not is_overriding:
            logging.warning(
                "Delayed connection %s is not overriding any weight=0 Connection",
                conn_ctx.conf["Name"],
            )


def _display_override_chains(
    processed_conn_blocks: list[_ConnCtx], config_assignment: re.Pattern[str]
) -> None:
    """Phase 3a: Display override chains for debugging purposes."""
    for conn_ctx in reversed(processed_conn_blocks):
        if conn_ctx.override and not conn_ctx.visited:
            logging.warning("Connection %s takes part in overriding chain:", conn_ctx.conf["Name"])
            current = conn_ctx
            while current:
                logging.info(
                    " -> %-6s %-60.60s Weight: %-8s SpontMinis: %-8s SynConfigure: %s",
                    "(base)" if not current.override else " ^",
                    f"{current.conf['Name']}  ",
                    f"{current.conf['Source']} -> {current.conf['Destination']}",
                    current.conf.get("Weight", "-"),
                    current.conf.get("SpontMinis", "-"),
                    ", ".join(_get_syn_config_vars(current, config_assignment)),
                )
                if current.visited:
                    break
                current.visited = True
                current = current.override


def _check_global_vars(conn_configure_global_vars: dict[str, list[str]]) -> None:
    """Phase 3b: Warn if any SynapseConfigure uses global variables."""
    if conn_configure_global_vars:
        logging.warning(
            "Global variables in SynapseConfigure. Review the following "
            "connections and move the global vars to Conditions block"
        )
        for name, vars_ in conn_configure_global_vars.items():
            logging.warning(" -> %s: %s", name, vars_)
    else:
        logging.info(" => CHECK No Global vars!")


def _check_weight0_overrides(zero_weight_conns: list[_ConnCtx]) -> None:
    """Phase 3c: Warn about or raise errors for weight=0 overrides."""
    not_overridden_weight_0 = []
    for conn_ctx in zero_weight_conns:
        if conn_ctx.override is None:
            not_overridden_weight_0.append(conn_ctx)
        elif not conn_ctx.full_override:
            raise ConfigurationError(
                f"Partial Weight=0 override is not supported: Conn {conn_ctx.conf['Name']}"
            )

    if not_overridden_weight_0:
        logging.warning(
            "The following connections with Weight=0 are not overridden, "
            "thus won't be instantiated:"
        )
        for conn in not_overridden_weight_0:
            logging.warning(" -> %s", conn.conf["Name"])
    else:
        logging.info(" => CHECK No single Weight=0 blocks!")


def check_connections_configure(config: _SimConfig, target_manager) -> None:
    """Check connection block configuration and raise warnings for:
    1. Global variables should be set in the Conditions block
    2. Connection overriding chains (t=0)
    3. Connections with Weight=0 not overridden by delayed (not instantiated)
    4. Partial connection overriding -> Error
    5. Connections with delay > 0 not overriding anything
    """
    logging.info("Checking Connection Configurations")

    config_assignment = re.compile(r"(\S+)\s*\*?=\s*(\S+)")
    all_conn_blocks = [_ConnCtx(c) for c in config.connections.values()]
    processed_conn_blocks: list[_ConnCtx] = []
    zero_weight_conns: list[_ConnCtx] = []
    conn_configure_global_vars: dict[str, list[str]] = defaultdict(list)

    _build_override_chains(
        all_conn_blocks,
        processed_conn_blocks,
        zero_weight_conns,
        conn_configure_global_vars,
        config_assignment,
        target_manager,
    )
    _process_delayed_connections(all_conn_blocks, zero_weight_conns, target_manager)
    _display_override_chains(processed_conn_blocks, config_assignment)
    _check_global_vars(conn_configure_global_vars)
    _check_weight0_overrides(zero_weight_conns)


def _input_resistance(config: _SimConfig, target_manager):
    prop = "@dynamics:input_resistance"
    for stim in config.stimuli:
        target_or_cs_name = stim["Target"] if "Target" in stim else stim["CompartmentSet"]
        if stim["Mode"] == "Conductance" and stim["Pattern"] in {
            "RelativeShotNoise",
            "RelativeOrnsteinUhlenbeck",
        }:
            # NOTE: use target_manager to read the population names of hoc or NodeSet targets
            population_names = (
                target_manager.get_target(target_or_cs_name).population_names
                if "Target" in stim
                else [target_manager.get_compartment_set(target_or_cs_name).population]
            )
            for population in population_names:
                config._cell_requirements.setdefault(population, set()).add(prop)
                log_verbose(f"[cell] {prop} ({population}:{target_or_cs_name})")
