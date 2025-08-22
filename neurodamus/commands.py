"""Module implementing entry functions"""

import logging
import os
import sys
import time
from pathlib import Path

from docopt import docopt

from .core import MPI, OtherRankError
from .core.configuration import EXCEPTION_NODE_FILENAME, ConfigurationError, LogLevel
from .utils.pyutils import docopt_sanitize
from neurodamus.node import Neurodamus
from neurodamus.utils.timeit import TimerManager


def neurodamus(args=None):
    """Neurodamus

    Usage:
        neurodamus <ConfigFile> [options]
        neurodamus --version
        neurodamus --help

    Options:
        -v --verbose            Increase verbosity level
        --debug                 Extremely verbose mode for debugging
        --disable-reports       Disable all reports [default: False]
        --build-model=[AUTO, ON, OFF]
                                Shall it build and eventually overwrite? [default: AUTO]
                                - AUTO: build model if doesn't exist and simulator is coreneuron
                                - ON: always build and eventually overwrite the model
                                - OFF: Don't build the model. Simulation may fail to start
        --simulate-model=[ON, OFF]
                                Shall the simulation start automatically? [default: ON]
        --output-path=PATH      Alternative output directory, overriding the config file's
        --keep-build            Keep coreneuron intermediate data in a folder named `build`.
                                Otherwise deleted at the end. ``--save=<PATH>`` overrides this.
        --modelbuilding-steps=<number>
                                Set the number of ModelBuildingSteps for the CoreNeuron sim
        --experimental-stims    Shall use only Python stimuli? [default: False]
        --lb-mode=[RoundRobin, WholeCell, MultiSplit, Memory]
                                The Load Balance mode.
                                - RoundRobin: Disable load balancing. Good for quick simulations
                                - WholeCell: Does a first pass to compute load balancing and
                                    redistributes cells so that CPU load is similar among ranks
                                - MultiSplit: Allows splitting cells into pieces for distribution.
                                    WARNING: This mode is incompatible with CoreNeuron
                                - Memory: Load balance based on memory usage. By default, it uses
                                    the "allocation_r#_c#.pkl.gz" file to load a pre-computed load
                                    balance
        --save=<PATH>           Path to create a save point (at tstop) to enable restore. Only
                                available for CoreNEURON.
        --restore=<PATH>        Restore and resume simulation from a save point. Only available
                                for CoreNEURON.
        --dump-cell-state=<GID(s)>
                                Dump cell state debug files on tstart and tstop.
                                For NEURON, accepts a list of GIDs or ranges (e.g., 1,2,3-6,9).
                                For CoreNEURON, behavior is unchanged and only one GID is accepted.
                                If a list is provided, only the first GID will be used.
        --enable-shm=[ON, OFF]  Enables the use of /dev/shm for coreneuron_input (available
                                only on linux) [default: OFF]
        --model-stats           Show model stats in CoreNEURON simulations [default: False]
        --dry-run               Dry-run simulation to estimate memory usage [default: False]
        --crash-test            Run the simulation with single section cells and single synapses
        --num-target-ranks=<number>  Number of ranks to target for dry-run load balancing
        --coreneuron-direct-mode     Run CoreNeuron in direct memory mode transfered from Neuron,
                                     without writing model data to disk.
        --use-color=[ON, OFF]  If OFF, forces no color to be used in logs; [default: ON]
        --report-buffer-size=<number> Override the size in MB each rank will allocate for each
                                      report buffer to hold data. When the buffer is full, the
                                      ranks will aggregate data for writing to disk. Default: 8 MB
    """
    from . import __version__

    options = docopt_sanitize(docopt(neurodamus.__doc__, args, version=__version__))
    config_file = options.pop("ConfigFile")
    log_level = _pop_log_level(options)

    if not os.path.isfile(config_file):
        logging.error("Config file not found: %s", config_file)
        return 1

    # Shall replace process with special? Don't if is special or already replaced
    if not sys.argv[0].endswith("special") and not os.environ.get("NEURODAMUS_SPECIAL"):
        _attempt_launch_special(config_file)

    # Warning control before starting the process
    _filter_warnings()

    # Some previous executions may have left a bad exception node file
    # This is done now so it's a very early stage and we know the mpi rank
    if MPI.rank == 0 and os.path.exists(EXCEPTION_NODE_FILENAME):
        os.remove(EXCEPTION_NODE_FILENAME)

    try:
        Neurodamus(config_file, auto_init=True, logging_level=log_level, **options).run()
        TimerManager.timeit_show_stats()
    except ConfigurationError:  # Common, only show error in Rank 0
        if MPI._rank == 0:  # Use _rank so that we avoid init
            logging.exception("ConfigurationError")
        return 1
    except OtherRankError:
        return 1  # no need for _mpi_abort, error is being handled by all ranks
    except:  # noqa: E722
        show_exception_abort("Unhandled Exception. Terminating...", sys.exc_info())
        return 1
    return 0


def _pop_log_level(options):
    log_level = LogLevel.DEFAULT
    if options.pop("debug", False):
        log_level = LogLevel.DEBUG
    elif options.pop("verbose", False):
        log_level = LogLevel.VERBOSE

    if log_level >= LogLevel.VERBOSE:
        from pprint import pprint

        pprint(options)  # noqa: T203

    return log_level


def show_exception_abort(err_msg, exc_info):
    """Show an exception info in only one rank

    Several ranks are likely to be in sync so a simple touch wont work.
    Processes that dont see any file will register (append) their rank id
    First one is elected to print
    """
    err_file = Path(EXCEPTION_NODE_FILENAME)
    ALL_RANKS_SYNC_WINDOW = 1

    if err_file.exists():
        return

    with open(err_file, "a", encoding="utf-8") as f:
        f.write(str(MPI.rank) + "\n")

    with open(err_file, encoding="utf-8") as f:
        line0 = f.readline().strip()

    if str(MPI.rank) == line0:
        logging.critical(err_msg, exc_info=exc_info)

    time.sleep(ALL_RANKS_SYNC_WINDOW)  # give time to the rank that logs the exception
    _mpi_abort()  # abort all ranks which have waited. Seems to help avoiding MPT stack


def _attempt_launch_special(config_file):
    import shutil

    special = shutil.which("special")
    if os.path.isfile("x86_64/special"):  # prefer locally compiled special
        special = os.path.abspath("x86_64/special")
    if special is None:
        logging.warning(
            "special not found. Running neurodamus from Python with libnrnmech. "
            "-> DO NOT USE WITH PRODUCTION RUNS"
        )
        return
    neurodamus_py_root = os.environ.get("NEURODAMUS_PYTHON")
    if not neurodamus_py_root:
        logging.warning(
            "No NEURODAMUS_PYTHON set. Running neurodamus from Python with libnrnmech. "
            "-> DO NOT USE WITH PRODUCTION RUNS"
        )
        return
    print("::INIT:: Special available. Replacing binary...")  # noqa: T201
    os.environ["NEURODAMUS_SPECIAL"] = "1"
    init_script = os.path.join(neurodamus_py_root, "init.py")
    os.execl(special, "-mpi", "-python", init_script, "--configFile=" + config_file, *sys.argv[2:])


def _mpi_abort():
    import ctypes

    c_api = ctypes.CDLL(None)
    c_api.MPI_Abort(0)


def _filter_warnings():
    """Control matched warning to display once in rank 0.

    Warning 1:
    "special" binaries built with %intel build_type=Release,RelWithDebInfo flushes
    denormal results to zero, which triggers the numpy warning for subnormal in every rank.
    Reduce this type of warning displayed once in rank0.
    Note: "special" with build_type = FastDebug/Debug or calling the simulation process
       in python (built with gcc) does not have such flush-to-zero warning.
    """
    import warnings

    action = "once" if MPI.rank == 0 else "ignore"

    warnings.filterwarnings(
        action=action,
        message="The value of the smallest subnormal for .* type is zero.",
        category=UserWarning,
        module="numpy",
    )
