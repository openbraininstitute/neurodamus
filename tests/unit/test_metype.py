from unittest.mock import Mock, seal
import pytest

from itertools import product

from neurodamus.metype import SectionIdError, BaseCell

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
            offset = flags[0]*10
            mock_cell.CellRef.node = list(range(32+offset, 42+offset))
            mock_cell.CellRef.nSecNodal = 10
        if flags[2]:
            offset = flags[0]*10 + flags[1]*10
            mock_cell.CellRef.myelin = list(range(32+offset, 42+offset))
            mock_cell.CellRef.nSecMyelinated = 10
        seal(mock_cell.CellRef)

        assert mock_cell.get_section_id("TestCell[0].soma[0]") == 0 * 10
        assert mock_cell.get_section_id("TestCell[0].axon[1]") == 1 * 10 + 1
        assert mock_cell.get_section_id("TestCell[0].dend[2]") == 2 * 10 + 2
        assert mock_cell.get_section_id("TestCell[0].apic[3]") == 3 * 10 + 3
        if flags[0]:
            mock_cell.get_section_id("TestCell[0].ais[4]") == 4 * 10 + 4
        else:
            with pytest.raises(SectionIdError):
                mock_cell.get_section_id("TestCell[0].ais[4]")
        if flags[1]:
            mock_cell.get_section_id("TestCell[0].node[5]") == 4 * 10 + 5 + flags[0]*10
        else:
            with pytest.raises(SectionIdError):
                mock_cell.get_section_id("TestCell[0].node[5]")
        if flags[2]:
            mock_cell.get_section_id("TestCell[0].myelin[6]") == 4 * 10 + 6 + flags[0]*10 + flags[1]*10
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