import logging
import os
from pathlib import Path

from . import NeuronWrapper as Nd
from ._utils import run_only_rank0
from .configuration import ConfigurationError, SimConfig


class CompartmentMapping:
    """Interface to register section segment mapping with NEURON."""

    def __init__(self, cell_distributor):
        self.cell_distributor = cell_distributor
        self.pc = Nd.ParallelContext()

    @staticmethod
    def create_section_vectors(section_id, section, secvec, segvec):
        num_segments = 0
        for seg in section:
            secvec.append(section_id)
            segvec.append(seg.node_index())
            num_segments += 1

        return num_segments

    def process_section(self, cell, section, num_electrodes, all_lfp_factors, section_offset):
        secvec, segvec, lfp_factors = Nd.Vector(), Nd.Vector(), Nd.Vector()
        num_segments = 0
        section_attr = getattr(cell._cellref, section[0], None)
        if section_attr:
            for sec in section_attr:
                section_id = cell.get_section_id(sec)
                num_segments += self.create_section_vectors(section_id, sec, secvec, segvec)

        if num_electrodes > 0 and all_lfp_factors.size() > 0 and num_segments > 0:
            start_idx = section_offset * num_electrodes
            end_idx = (section_offset + num_segments) * num_electrodes - 1
            lfp_factors.copy(all_lfp_factors, start_idx, end_idx)

        self.pc.nrnbbcore_register_mapping(
            cell.gid, section[1], secvec, segvec, lfp_factors, num_electrodes
        )
        return num_segments

    def register_mapping(self):
        sections = [
            ("somatic", "soma"),
            ("axonal", "axon"),
            ("basal", "dend"),
            ("apical", "apic"),
            ("AIS", "ais"),
            ("nodal", "node"),
            ("myelinated", "myelin"),
        ]
        gidvec = self.cell_distributor.getGidListForProcessor()
        for activegid in gidvec:
            cell = self.cell_distributor.get_cell(activegid)
            all_lfp_factors = Nd.Vector()
            num_electrodes = 0
            lfp_manager = getattr(self.cell_distributor, "_lfp_manager", None)
            if lfp_manager:
                pop_info = self.cell_distributor.getPopulationInfo(activegid)
                num_electrodes = lfp_manager.get_number_electrodes(activegid, pop_info)
                all_lfp_factors = lfp_manager.read_lfp_factors(activegid, pop_info)

            section_offset = 0
            for section in sections:
                processed_segments = self.process_section(
                    cell, section, num_electrodes, all_lfp_factors, section_offset
                )
                section_offset += processed_segments


class _CoreNEURONConfig:
    """Responsible for managing the configuration of the CoreNEURON simulation.

    It writes the simulation / report configurations and calls the CoreNEURON solver.

    Note: this creates the `CoreConfig` singleton
    """

    default_cell_permute = 0
    artificial_cell_object = None

    @property
    def sim_config_file(self):
        """Get sim config file path to be saved"""
        return str(Path(self.build_path) / "sim.conf")

    @property
    def report_config_file_save(self):
        """Get report config file path to be saved"""
        return str(Path(self.build_path) / "report.conf")

    @property
    def report_config_file_restore(self):
        """Get report config file path to be restored

        We need this file and path for restoring because we cannot recreate it
        from scratch. Only usable when restore exists and is a dir
        """
        return str(Path(SimConfig.restore) / "report.conf")

    @property
    def output_root(self):
        """Get output root from SimConfig"""
        return SimConfig.output_root

    @property
    def datadir(self):
        """Get datadir from SimConfig if not set explicitly"""
        return SimConfig.coreneuron_datadir_path()

    @property
    def build_path(self):
        """Save root folder"""
        return SimConfig.build_path()

    @property
    def restore_path(self):
        """Restore root folder"""
        return SimConfig.restore

    # Instantiates the artificial cell object for CoreNEURON
    # This needs to happen only when CoreNEURON simulation is enabled
    def instantiate_artificial_cell(self):
        self.artificial_cell_object = Nd.CoreNEURONArtificialCell()

    @run_only_rank0
    def update_report_config(self, substitutions):
        """Updates a report configuration (e.g., stop time).

        Searches for the specified report and nodeset, updates the relevant parameters
        (currently only `tstop`), and writes the updated configuration to a new file.

        Note: `report.conf` must already exist.
        """
        report_conf = Path(self.report_config_file_save)

        # Read all content
        with report_conf.open("rb") as f:
            lines = f.readlines()

        # Track performed substitutions
        applied_subs = set()

        # Find and update the matching line
        for i, line in enumerate(lines):
            try:
                parts = line.decode().split()
                key = tuple(parts[0:2])  # Report name and target name

                if key in substitutions:
                    # This is often but not always tstop:
                    # new_tend = min(tstop, tend) where tend is the ending
                    # of the report and tstop is the tstop of this simulation
                    # (potentially between a restore and a save)
                    new_tend = substitutions[key]
                    parts[9] = f"{new_tend:.6f}"
                    lines[i] = (" ".join(parts) + "\n").encode()
                    applied_subs.add(key)
            except (UnicodeDecodeError, IndexError):  # noqa: PERF203
                # Ignore lines that cannot be decoded (binary data)
                continue

        # Find substitutions that were not applied
        missing_subs = set(substitutions.keys()) - applied_subs

        if missing_subs:
            raise ConfigurationError(
                f"Some substitutions could not be applied for the following "
                f"(report, target) pairs: {missing_subs}"
            )

        with report_conf.open("wb") as f:
            f.writelines(lines)

    @run_only_rank0
    def write_report_config(
        self,
        report_name,
        target_name,
        report_type,
        report_variable,
        unit,
        report_format,
        target_type,
        dt,
        start_time,
        end_time,
        gids,
        buffer_size=8,
    ):
        """Here we append just one report entry to report.conf. We are not writing the full file as
        this is done incrementally in Node.enable_reports
        """
        import struct

        num_gids = len(gids)
        logging.info("Adding report %s for CoreNEURON with %s gids", report_name, num_gids)
        report_conf = Path(self.report_config_file_save)
        report_conf.parent.mkdir(parents=True, exist_ok=True)
        with report_conf.open("ab") as fp:
            # Write the formatted string to the file
            fp.write(
                (
                    "%s %s %s %s %s %s %d %lf %lf %lf %d %d\n"  # noqa: UP031
                    % (
                        report_name,
                        target_name,
                        report_type,
                        report_variable,
                        unit,
                        report_format,
                        target_type,
                        dt,
                        start_time,
                        end_time,
                        num_gids,
                        buffer_size,
                    )
                ).encode()
            )
            # Write the array of integers to the file in binary format
            fp.write(struct.pack(f"{num_gids}i", *gids))
            fp.write(b"\n")

    @run_only_rank0
    def write_sim_config(
        self,
        tstop: float,
        dt: float,
        prcellgid: int,
        celsius: float,
        v_init: float,
        pattern=None,
        seed=None,
        model_stats=False,
        enable_reports=True,
    ):
        """Writes the simulation configuration to a file.

        Args:
            tstop (float): Simulation stop time.
            dt (float): Time step for the simulation.
            prcellgid (int): dump cell state GID. CoreNeuron allows only one
                cell to be dumped at a time.
            celsius (float): Temperature in Celsius.
            v_init (float): Initial voltage.
            pattern (str, optional): Pattern for the simulation. Defaults to None.
            seed (int, optional): Random seed for the simulation. Defaults to None.
            model_stats (bool, optional): Flag to enable model statistics. Defaults to False.
            enable_reports (bool, optional): Flag to enable reports. Defaults to True.
        """
        simconf = Path(self.sim_config_file)
        logging.info("Writing sim config file: %s", simconf)
        simconf.parent.mkdir(parents=True, exist_ok=True)

        with simconf.open("w", encoding="utf-8") as fp:
            fp.write(f"outpath='{os.path.abspath(self.output_root)}'\n")
            fp.write(f"datpath='{os.path.abspath(self.datadir)}'\n")
            fp.write(f"tstop={tstop}\n")
            fp.write(f"dt={dt}\n")
            fp.write(f"prcellgid={prcellgid}\n")
            fp.write(f"celsius={celsius}\n")
            fp.write(f"voltage={v_init}\n")
            fp.write(f"cell-permute={int(self.default_cell_permute)}\n")
            if pattern:
                fp.write(f"pattern='{pattern}'\n")
            if seed:
                fp.write(f"seed={int(seed)}\n")
            if model_stats:
                fp.write("'model-stats'\n")
            if enable_reports:
                fp.write(f"report-conf='{self.report_config_file_save}'\n")
            fp.write(f"mpi={os.environ.get('NEURON_INIT_MPI', '1')}\n")

        logging.info(" => Dataset written to '%s'", simconf)

    @run_only_rank0
    def write_report_count(self, count, mode="w"):
        report_config = Path(self.report_config_file_save)
        report_config.parent.mkdir(parents=True, exist_ok=True)
        with report_config.open(mode) as fp:
            fp.write(f"{count}\n")

    @run_only_rank0
    def write_population_count(self, count):
        self.write_report_count(count, mode="a")

    @run_only_rank0
    def write_spike_population(self, population_name, population_offset=None):
        report_config = Path(self.report_config_file_save)
        report_config.parent.mkdir(parents=True, exist_ok=True)
        with report_config.open("a", encoding="utf-8") as fp:
            fp.write(population_name)
            if population_offset is not None:
                fp.write(f" {int(population_offset)}")
            fp.write("\n")

    @run_only_rank0
    def write_spike_filename(self, filename):
        report_config = Path(self.report_config_file_save)
        report_config.parent.mkdir(parents=True, exist_ok=True)
        with report_config.open("a", encoding="utf-8") as fp:
            fp.write(filename)
            fp.write("\n")

    def psolve_core(self, coreneuron_direct_mode=False):
        from neuron import coreneuron

        from . import NeuronWrapper as Nd

        Nd.cvode.cache_efficient(1)
        coreneuron.enable = True
        coreneuron.file_mode = not coreneuron_direct_mode
        coreneuron.sim_config = f"{self.sim_config_file}"
        # set build_path only if the user explicitly asked with --save
        # in this way we do not create 1_2.dat and time.dat if not needed
        if SimConfig.save:
            coreneuron.save_path = self.build_path
        if SimConfig.restore:
            coreneuron.restore_path = self.restore_path

        # Model is already written to disk by calling pc.nrncore_write()
        coreneuron.skip_write_model_to_disk = True
        coreneuron.model_path = f"{self.datadir}"
        Nd.pc.psolve(Nd.tstop)


# Singleton
CoreConfig = _CoreNEURONConfig()
