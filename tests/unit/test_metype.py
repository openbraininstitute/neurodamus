from itertools import product
from unittest.mock import Mock, seal

import numpy.testing as npt
import pytest

from tests.conftest import RINGTEST_DIR

from neurodamus import Node
from neurodamus.metype import BaseCell, Cell_V6, SectionIdError


def _make_mock_cell():
    mock_cell_ref = Mock()
    mock_cell_ref.soma = list(range(10))
    mock_cell_ref.axon = [10, 11]
    mock_cell_ref.nSecAxonalOrig = 10
    mock_cell_ref.dend = list(range(12, 22))
    mock_cell_ref.apic = list(range(22, 32))

    mock_cell = BaseCell()
    mock_cell._cellref = mock_cell_ref

    return mock_cell


def test_get_section_id():
    """
    Test the `get_section_id` function for all valid combinations of optional sections.

    This test generates all 8 possible combinations of presence/absence for the
    optional sections: 'ais', 'node', and 'myelin'. Each combination is controlled
    by a 3-boolean flag tuple generated via `itertools.product`.

    For each configuration:
    - A mock cell is constructed with soma, axon, dend, and apic always present.
    - Optional sections (ais, node, myelin) are added conditionally based on the flags.
    - Section IDs are checked for correctness or failure (if the section is absent).
    - Offsets are validated against the order defined in `_section_layout`.

    This ensures `get_section_id` correctly handles both presence and absence of
    sections and computes the global index accurately.
    """
    for flags in product([False, True], repeat=3):
        mock_cell = _make_mock_cell()
        if flags[0]:
            mock_cell.CellRef.ais = list(range(32, 42))
            mock_cell.CellRef.nSecLastAIS = 10
        if flags[1]:
            offset = flags[0] * 10
            mock_cell.CellRef.node = list(range(32 + offset, 42 + offset))
            mock_cell.CellRef.nSecNodal = 10
        if flags[2]:
            offset = flags[0] * 10 + flags[1] * 10
            mock_cell.CellRef.myelin = list(range(32 + offset, 42 + offset))
            mock_cell.CellRef.nSecMyelinated = 10
        seal(mock_cell.CellRef)

        assert mock_cell.get_section_id("TestCell[0].soma[0]") == 0 * 10
        assert mock_cell.get_section_id("TestCell[0].axon[1]") == 1 * 10 + 1
        assert mock_cell.get_section_id("TestCell[0].dend[2]") == 2 * 10 + 2
        assert mock_cell.get_section_id("TestCell[0].apic[3]") == 3 * 10 + 3
        if flags[0]:
            assert mock_cell.get_section_id("TestCell[0].ais[4]") == 4 * 10 + 4
        else:
            with pytest.raises(SectionIdError):
                mock_cell.get_section_id("TestCell[0].ais[4]")
        if flags[1]:
            assert mock_cell.get_section_id("TestCell[0].node[5]") == 4 * 10 + 5 + flags[0] * 10
        else:
            with pytest.raises(SectionIdError):
                mock_cell.get_section_id("TestCell[0].node[5]")
        if flags[2]:
            assert (
                mock_cell.get_section_id("TestCell[0].myelin[6]")
                == 4 * 10 + 6 + flags[0] * 10 + flags[1] * 10
            )
        else:
            with pytest.raises(SectionIdError):
                mock_cell.get_section_id("TestCell[0].myelin[6]")


def test_get_sec():
    """
    Test the `get_sec` function to ensure correct section retrieval by global index.

    This test:
    - Creates a mock cell with predefined soma, axon, dend, and apic sections.
    - Verifies that valid global indices return the expected local section indices.
    - Confirms that out-of-range indices raise a `SectionIdError`.

    Specifically:
    - Indices within soma, axon, dend, and apic return their corresponding local indices.
    - Indices beyond defined sections correctly raise exceptions.
    """
    mock_cell = _make_mock_cell()
    seal(mock_cell.CellRef)
    assert mock_cell.get_sec(9) == 9

    assert mock_cell.get_sec(10) == 10
    assert mock_cell.get_sec(11) == 11
    with pytest.raises(SectionIdError):
        mock_cell.get_sec(12)
    assert mock_cell.get_sec(20) == 12
    assert mock_cell.get_sec(21) == 13
    with pytest.raises(SectionIdError):
        mock_cell.get_sec(1111)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {"network": str(RINGTEST_DIR / "circuit_config_bigA.json")},
        }
    ],
    indirect=True,
)
def test_get_segment_points(create_tmp_simulation_config_file):
    """Test the function Cell_V6:get_all_segment_points"""
    n = Node(create_tmp_simulation_config_file, {"enable_coord_mapping": True})
    n.load_targets()
    n.create_cells()
    cell_manager = n.circuits.get_node_manager("RingA")
    assert len(cell_manager.cells) == 3
    cell0 = next(iter(cell_manager.cells))
    assert isinstance(cell0, Cell_V6)
    assert cell0.gid == 0
    cell0.compute_segment_global_coordinates()
    assert len(cell0.segment_global_coords) == 11
    soma_obj = cell0.CellRef.soma[0]
    soma_seg_points = cell0.segment_global_coords[soma_obj.name()]
    assert len(soma_seg_points) == soma_obj.nseg + 1
    npt.assert_allclose(
        soma_seg_points,
        [
            [-31.542698, -0.650061, 2.0],
            [-10.916599, 2.58374, 2.0],
            [9.709499, 5.817541, 2.0],
            [30.335598, 9.051342, 2.0],
        ],
    )
    dend0 = cell0.CellRef.dend[0]
    dend0_seg_points = cell0.segment_global_coords[dend0.name()]
    assert len(dend0_seg_points) == dend0.nseg + 1
    npt.assert_allclose(dend0_seg_points, [[15.0, 1.0, 2.0], [115.0, 1.0, 2.0], [215.0, 1.0, 2.0]])
