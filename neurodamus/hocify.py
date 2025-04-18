import logging
import multiprocessing
import os
from time import strftime

from .core.cell import Cell
from .utils.logging import log_stage, log_verbose, setup_logging
from .utils.progressbar import ProgressBar

FASTHOC_DIRNAME = "_fasthoc"


def process_file(file_tuple):
    src_file, dst_file = file_tuple
    try:
        c = Cell()
        c.load_morphology(src_file, export_commands=True)
        with open(dst_file, "w") as f:
            for cmd in c._commands:
                if hasattr(cmd, "s"):
                    f.write(cmd.s.removeprefix("~"))
                else:
                    f.write(cmd + "\n")
    except Exception as e:
        e.args = ("Processing " + src_file, *e.args)
        return e
    return src_file


class Hocify:
    fasthoclogfile = "hocify-{}.log".format(strftime("%Y-%m-%d_%Hh%M"))

    def __init__(self, morpho_path, neuron_nframe, log_level, output_dir, **_user_opts):
        Hocify.fasthocdir = output_dir or os.path.join(morpho_path, FASTHOC_DIRNAME)
        os.mkdir(Hocify.fasthocdir)

        Hocify.fasthoclogfile = os.path.join(Hocify.fasthocdir, Hocify.fasthoclogfile)
        setup_logging(loglevel=log_level, logfile=Hocify.fasthoclogfile, rank=0)

        log_stage("Initializing.  Logfile: " + Hocify.fasthoclogfile)
        self._morpho_path = morpho_path
        logging.info("Morphology path set to: %s", self._morpho_path)

        Hocify.nframe = neuron_nframe
        os.environ["NEURON_NFRAME"] = str(Hocify.nframe)
        logging.info("NEURON_NFRAME set to: %s", Hocify.nframe)

    def convert(self, morpho_folder="ascii"):
        log_stage("Starting conversion")
        self._morphdir = os.path.join(self._morpho_path, morpho_folder)
        if not os.path.isdir(self._morphdir):
            logging.critical("Morphology directory does not exist: %s", self._morphdir)
            return

        logging.info("Target fast hoc folder is: %s", Hocify.fasthocdir)
        logging.info("Hoc-ifying morphology folder: %s ...", self._morphdir)

        self._target_morph_files = (
            (
                os.path.join(self._morphdir, f.name),
                os.path.join(Hocify.fasthocdir, f.name.replace(".asc", ".hoc")),
            )
            for f in os.scandir(self._morphdir)
            if f.is_file() and f.name.endswith(".asc")
        )

        pool = multiprocessing.Pool()
        processed_files = pool.imap_unordered(process_file, self._target_morph_files)
        estimated_file_count = os.lstat(self._morphdir).st_nlink - 2  # ok for large dirs
        for file in ProgressBar.iter(processed_files, estimated_file_count):
            if isinstance(file, Exception):
                logging.critical(str(file))
                return
            log_verbose("Done for: " + file)
        logging.info("Done")
