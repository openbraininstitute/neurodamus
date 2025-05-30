import os
from pathlib import Path

import psutil

from neurodamus.utils.pyutils import rmtree

SHM = "/dev/shm"  # noqa: S108


class SHMUtil:
    """Helper class for the SHM file transfer mechanism of CoreNEURON."""

    node_id = -1
    nnodes = -1

    @staticmethod
    def _set_node_info():  # TODO: Replace with MPI SHM communicator
        from neurodamus.core import MPI

        shmdir = SHMUtil.get_datadir_shm("/.__pydamus_nodeinfo_sync")
        path = os.path.join(shmdir, str(MPI.rank))

        # Create SHM folder and files to sync the process count per node
        try:
            os.makedirs(shmdir, exist_ok=True)
            os.close(os.open(path, os.O_CREAT))
        except FileExistsError:
            pass  # Ignore if we have already created the files

        MPI.barrier()

        # Get a filelist sorted by rank ID and store the local node info
        listdir = sorted(os.listdir(shmdir), key=int)
        rank0_node = int(listdir[0])
        nranks_node = len(listdir)

        # Calculate node ID based on the entries that contain a process count
        node_info = MPI.py_gather((nranks_node if MPI.rank == rank0_node else 0), 0)
        if MPI.rank == 0:
            node_info[0] = 0
            for i in range(1, MPI.size):
                node_info[i] = node_info[i - 1] + int(node_info[i] > 0)

        # At this point, the node ID for the rank and # nodes are determined
        SHMUtil.node_id = MPI.py_scatter(node_info, 0)
        SHMUtil.nnodes = MPI.py_broadcast((node_info[-1] + 1) if MPI.rank == 0 else 0, 0)

        rmtree(shmdir)

    @staticmethod
    def _get_approximate_node_rss():
        """Reliable, but imprecise estimate of the nodewide RSS.

        Note: This estimate must work even if MPI ranks can't be associated with
        a physical node.
        """
        vm = psutil.virtual_memory()
        return vm.total - vm.available

    @staticmethod
    def is_node_id_known():
        """Can MPI ranks be associated with physical nodes?"""
        return SHMUtil.get_datadir_shm() is not None

    @staticmethod
    def get_nodewise_rss():
        """For each node the sum of the RSS of all MPI ranks on that node.

        If MPI ranks can't be associated with a node, return `None`.
        """
        from neurodamus.core import MPI, Neuron

        # If we do not have the SHM environment, we can't even know
        # how many nodes there are. Just return `None`.
        if not SHMUtil.is_node_id_known():
            return None

        # Define the node ID for the rank and number of nodes
        if SHMUtil.nnodes < 0:
            SHMUtil._set_node_info()

        # Aggregate the individual memory consumption per node
        process = psutil.Process(os.getpid())
        rss = Neuron.Vector(SHMUtil.nnodes, 0.0)
        rss[SHMUtil.node_id] = process.memory_info().rss

        MPI.allreduce(rss, MPI.SUM)

        return rss.as_numpy().copy()

    @staticmethod
    def get_node_rss():
        # If we do not have the SHM environment, ignore and return an estimate
        if not SHMUtil.is_node_id_known():
            return SHMUtil._get_approximate_node_rss()

        # Return the consumption estimated for the node
        rss = SHMUtil.get_nodewise_rss()
        return rss[SHMUtil.node_id]

    @staticmethod
    def get_mem_avail():
        return psutil.virtual_memory().available

    @staticmethod
    def get_mem_total():
        return psutil.virtual_memory().total

    @staticmethod
    def get_shm_avail():
        return psutil.disk_usage(SHM).free

    @staticmethod
    def get_datadir_shm(datadir=""):
        shmdir = os.environ.get("SHMDIR")
        if not shmdir or not shmdir.startswith(SHM):
            return None

        return str(Path(shmdir) / Path(datadir).resolve().relative_to("/"))

    @staticmethod
    def get_shm_factor():
        factor = os.environ.get("NEURODAMUS_SHM_FACTOR")
        return 0.4 if not factor or not 0.0 <= float(factor) <= 1.0 else float(factor)
