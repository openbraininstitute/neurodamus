"""Tests load balance."""

import logging
import re
import shutil
from pathlib import Path

import pytest

from .conftest import RINGTEST_DIR
from neurodamus.core.configuration import ConfigurationError, LoadBalanceMode

base_dir = Path("sim_conf")
pattern = "_loadbal_*.RingA"  # Matches any hash for population RingA


@pytest.fixture
def target_manager():
    from neurodamus.core.nodeset import NodeSet
    from neurodamus.target_manager import NodesetTarget

    nodes_t1 = NodeSet([1, 2, 3]).register_global("RingA")
    nodes_t2 = NodeSet([1]).register_global("RingA")
    t1 = NodesetTarget("All", [nodes_t1], [nodes_t1])
    t2 = NodesetTarget("VerySmall", [nodes_t2], [nodes_t2])
    return MockedTargetManager(t1, t2)


def test_loadbalance_mode():
    assert LoadBalanceMode.parse("RoundRobin") == LoadBalanceMode.RoundRobin
    assert LoadBalanceMode.parse("WholeCell") == LoadBalanceMode.WholeCell
    assert LoadBalanceMode.parse("MultiSplit") == LoadBalanceMode.MultiSplit
    assert LoadBalanceMode.parse("Memory") == LoadBalanceMode.Memory
    with pytest.raises(ConfigurationError, match=r"Unknown load balance mode"):
        assert LoadBalanceMode.parse("Random")


def test_loadbal_no_cx(target_manager, caplog):
    from neurodamus.cell_distributor import LoadBalance, TargetSpec

    lbal = LoadBalance(1, "/gpfs/fake_path_to_nodes_1", "pop", target_manager, 4)
    assert not lbal._cx_targets
    assert not lbal._valid_loadbalance
    with caplog.at_level(logging.INFO):
        assert not lbal._cx_valid(TargetSpec("random_target"))
        assert " => No complexity files for current circuit yet" in caplog.records[-1].message


def test_loadbal_subtarget(target_manager, caplog):
    """Ensure given the right files are in the lbal dir, the correct situation is detected"""
    from neurodamus.cell_distributor import LoadBalance, TargetSpec

    nodes_file = "/gpfs/fake_node_path"
    lbdir, _ = LoadBalance._get_circuit_loadbal_dir(nodes_file, "RingA")
    shutil.copyfile(RINGTEST_DIR / "cx_RingA_RingA#.dat", lbdir / "cx_RingA_All#.dat")

    lbal = LoadBalance(1, nodes_file, "RingA", target_manager, 4)
    assert "RingA_All" in lbal._cx_targets
    assert not lbal._valid_loadbalance
    with caplog.at_level(logging.INFO):
        assert not lbal._cx_valid(TargetSpec("random_target"))
        assert " => No Cx files available for requested target" in caplog.records[-1].message
    assert lbal._cx_valid(TargetSpec("RingA:All"))  # yes!
    assert not lbal._cx_valid(TargetSpec("VerySmall"))  # not yet, need to derive subtarget

    with caplog.at_level(logging.INFO):
        assert lbal._reuse_cell_complexity(TargetSpec("RingA:VerySmall"))
        assert len(caplog.records) >= 2
        assert "Attempt reusing cx files from other targets..." in caplog.records[-2].message
        assert "Target VerySmall is a subset of the target RingA_All." in caplog.records[-1].message


@pytest.fixture
def circuit_conf_bigcell():
    """Test nodes file contains 1 big cell with 10 dendrites + 2 small cells with 2 dendrites"""
    from neurodamus.core.configuration import CircuitConfig

    circuit_base = str(RINGTEST_DIR)
    return CircuitConfig(
        CircuitPath=circuit_base,
        CellLibraryFile=circuit_base + "/nodes_A_bigcell.h5",
        METypePath=circuit_base + "/hoc",
        MorphologyPath=circuit_base + "/morphologies/asc",
        nrnPath="<NONE>",  # no connectivity
        CircuitTarget="All",
    )


@pytest.fixture
def circuit_conf():
    """Test nodes file contains 3 small cells with 2 dendrites each"""
    from neurodamus.core.configuration import CircuitConfig

    circuit_base = str(RINGTEST_DIR)
    return CircuitConfig(
        CircuitPath=circuit_base,
        CellLibraryFile=circuit_base + "/nodes_A.h5",
        METypePath=circuit_base + "/hoc",
        MorphologyPath=circuit_base + "/morphologies/asc",
        nrnPath="<NONE>",  # no connectivity
        CircuitTarget="All",
    )


def test_load_balance_integrated(target_manager, circuit_conf):
    """Comprehensive test using real cells and deriving cx for a sub-target"""
    from neurodamus.cell_distributor import CellDistributor, LoadBalance, TargetSpec

    cell_manager = CellDistributor(circuit_conf, target_manager)
    cell_manager.load_nodes()

    lbal = LoadBalance(1, circuit_conf.CircuitPath, "RingA", target_manager, 3)
    t1 = TargetSpec("RingA:All")
    assert not lbal._cx_valid(t1)

    with lbal.generate_load_balance(t1, cell_manager):
        cell_manager.finalize()

    assert "RingA_All" in lbal._cx_targets
    assert "RingA_All" in lbal._valid_loadbalance
    assert lbal._cx_valid(t1)

    # Check subtarget
    assert "RingA_VerySmall" not in lbal._cx_targets
    assert "RingA_VerySmall" not in lbal._valid_loadbalance
    assert lbal._reuse_cell_complexity(TargetSpec("RingA:VerySmall"))

    # Check not super-targets
    assert not lbal._reuse_cell_complexity(TargetSpec(None))


def test_MultiSplit_bigcell(target_manager, circuit_conf_bigcell, capsys):
    """Comprehensive test using ring circuit including one big cell,
    multi-split and complexity derivation
    """
    from neurodamus.cell_distributor import CellDistributor, LoadBalance, TargetSpec

    cell_manager = CellDistributor(circuit_conf_bigcell, target_manager)
    cell_manager.load_nodes()
    lbal = LoadBalance(
        LoadBalanceMode.MultiSplit, circuit_conf_bigcell.CircuitPath, "RingA", target_manager, 2
    )
    t1 = TargetSpec("RingA:All")
    assert not lbal._cx_valid(t1)

    with lbal.generate_load_balance(t1, cell_manager):
        cell_manager.finalize()

    captured = capsys.readouterr()
    assert "3 cells\n4 pieces" in captured.out
    assert re.match(
        r"(?s:.)*at least one cell is broken into 2 pieces \(bilist\[\d\], gid 1\)", captured.out
    )
    assert "RingA_All" in lbal._cx_targets
    assert "RingA_All" in lbal._valid_loadbalance
    assert lbal._cx_valid(t1)

    # Check the complexity file
    assert base_dir.exists()
    cx_filename = next(Path(".").glob(str(base_dir / pattern / "cx_RingA_All#.dat")))
    assert cx_filename.exists()
    with open(cx_filename) as cx_file:
        cx_saved = lbal._read_msdat(cx_file)
    assert list(cx_saved.keys()) == [1, 2, 3]
    assert float(cx_saved[1][0].split()[1]) > float(cx_saved[2][0].split()[1]) == float(
        cx_saved[3][0].split()[1]), (
        "cell complexity should be gid 1 > gid2 == gid3"
    )

    # Check the cpu assign file, format: ihost ngid (index, gid, subtreeindex) ..
    cpu_assign_filename = next(Path(".").glob(str(base_dir / pattern / "cx_RingA_All#.2.dat")))
    content = Path(cpu_assign_filename).open().read()
    assert content == "msgid 10000000\nnhost 2\n0 2  0 1 1  1 2 0\n1 2  0 1 0  2 3 0\n"

    # Ensure load-bal is reused for smaller targets in multisplit
    assert "RingA_VerySmall" not in lbal._cx_targets
    assert "RingA_VerySmall" not in lbal._valid_loadbalance
    assert lbal.valid_load_distribution(TargetSpec("RingA:VerySmall"))
    assert "RingA_VerySmall" in lbal._cx_targets
    assert "RingA_VerySmall" in lbal._valid_loadbalance
    captured = capsys.readouterr()
    assert "Target VerySmall is a subset of the target RingA_All" in captured.out


def test_MultiSplit(target_manager, circuit_conf, capsys):
    """Comprehensive test using ringtest cells, multi-split and complexity derivation"""
    from neurodamus.cell_distributor import CellDistributor, LoadBalance, TargetSpec

    cell_manager = CellDistributor(circuit_conf, target_manager)
    cell_manager.load_nodes()
    lbal = LoadBalance(
        LoadBalanceMode.MultiSplit, circuit_conf.CircuitPath, "RingA", target_manager, 2
    )
    t1 = TargetSpec("RingA:All")
    assert not lbal._cx_valid(t1)

    with lbal.generate_load_balance(t1, cell_manager):
        cell_manager.finalize()

    captured = capsys.readouterr()
    assert "3 cells\n3 pieces" in captured.out

    # Check the complexity file cx_RingA_All#.dat
    assert base_dir.exists()
    cx_filename = next(Path(".").glob(str(base_dir / pattern / "cx_RingA_All#.dat")))
    assert cx_filename.exists()
    with open(cx_filename) as cx_file:
        cx_saved = lbal._read_msdat(cx_file)
    assert list(cx_saved.keys()) == [1, 2, 3]
    assert float(cx_saved[1][0].split()[1]) == float(cx_saved[2][0].split()[1]) == float(
        cx_saved[3][0].split()[1]), (
        "cell complexity should be gid 1 == gid2 == gid3"
    )

    # Check the cpu assign file
    cpu_assign_filename = next(Path(".").glob(str(base_dir / pattern / "cx_RingA_All#.2.dat")))
    content = Path(cpu_assign_filename).open().read()
    assert content == "msgid 10000000\nnhost 2\n0 2  0 1 0  2 3 0\n1 1  1 2 0\n"


def test_WholeCell(target_manager, circuit_conf, capsys):
    """Ensure given the right files are in the lbal dir, the correct situation is detected"""
    from neurodamus.cell_distributor import CellDistributor, LoadBalance, TargetSpec

    cell_manager = CellDistributor(circuit_conf, target_manager)
    cell_manager.load_nodes()
    lbal = LoadBalance(
        LoadBalanceMode.WholeCell, circuit_conf.CircuitPath, "RingA", target_manager, 2
    )
    t1 = TargetSpec("RingA:All")
    assert not lbal._cx_valid(t1)

    with lbal.generate_load_balance(t1, cell_manager):
        cell_manager.finalize()

    captured = capsys.readouterr()
    assert "3 cells\n3 pieces" in captured.out

    # Check the cpu assign file
    cpu_assign_filename = next(Path(".").glob(str(base_dir / pattern / "cx_RingA_All#.2.dat")))
    content = Path(cpu_assign_filename).open().read()
    assert content == "msgid 10000000\nnhost 2\n0 2  0 1 0  2 3 0\n1 1  1 2 0\n"


def test_WholeCell_bigcell(target_manager, circuit_conf_bigcell, capsys):
    """Ensure given the right files are in the lbal dir, the correct situation is detected"""
    from neurodamus.cell_distributor import CellDistributor, LoadBalance, TargetSpec

    cell_manager = CellDistributor(circuit_conf_bigcell, target_manager)
    cell_manager.load_nodes()
    lbal = LoadBalance(
        LoadBalanceMode.WholeCell, circuit_conf_bigcell.CircuitPath, "RingA", target_manager, 2
    )
    t1 = TargetSpec("RingA:All")
    with lbal.generate_load_balance(t1, cell_manager):
        cell_manager.finalize()

    captured = capsys.readouterr()
    assert "3 cells\n3 pieces" in captured.out

    # Check the cpu assign file
    cpu_assign_filename = next(Path(".").glob(str(base_dir / pattern / "cx_RingA_All#.2.dat")))
    content = Path(cpu_assign_filename).open().read()
    assert content == "msgid 10000000\nnhost 2\n0 1  0 1 0\n1 2  1 2 0  2 3 0\n"


class MockedTargetManager:
    """
    A mock target manager, for the single purpose of returning the provided targets
    """

    def __init__(self, *targets) -> None:
        self.targets = {t.name.split(":")[-1]: t for t in targets}

    def get_target(self, target_spec, target_pop=None):
        from neurodamus.target_manager import TargetSpec

        if not isinstance(target_spec, TargetSpec):
            target_spec = TargetSpec(target_spec)
        if target_pop:
            target_spec.population = target_pop
        target_name = target_spec.name or TargetSpec.GLOBAL_TARGET_NAME
        target_pop = target_spec.population
        target = self.targets[target_name]
        return target if target_pop is None else target.make_subtarget(target_pop)

    def register_local_nodes(*_):
        pass
