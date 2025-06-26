from unittest.mock import Mock, seal
import pytest

from itertools import product

from neurodamus.metype import get_section_id, get_sec, SectionIdError

def _make_mock_cell():
    mock_cell = Mock()
    mock_cell.soma = list(range(10))
    mock_cell.nSecSoma = 10
    mock_cell.axon = [10, 11]
    mock_cell.nSecAxonalOrig = 10
    mock_cell.dend = list(range(12, 22))
    mock_cell.nSecBasal = 10
    mock_cell.apic = list(range(22, 32))
    mock_cell.nSecApical = 10
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
            mock_cell.ais = list(range(32, 42))
            mock_cell.nSecLastAIS = 10
        if flags[1]:
            offset = flags[0]*10
            mock_cell.node = list(range(32+offset, 42+offset))
            mock_cell.nSecNodal = 10
        if flags[2]:
            offset = flags[0]*10 + flags[1]*10
            mock_cell.myelin = list(range(32+offset, 42+offset))
            mock_cell.nSecMyelinated = 10
        seal(mock_cell)

        assert get_section_id(mock_cell, "TestCell[0].soma[0]") == 0 * 10
        assert get_section_id(mock_cell, "TestCell[0].axon[1]") == 1 * 10 + 1
        assert get_section_id(mock_cell, "TestCell[0].dend[2]") == 2 * 10 + 2
        assert get_section_id(mock_cell, "TestCell[0].apic[3]") == 3 * 10 + 3
        if flags[0]:
            get_section_id(mock_cell, "TestCell[0].ais[4]") == 4 * 10 + 4
        else:
            with pytest.raises(SectionIdError):
                get_section_id(mock_cell, "TestCell[0].ais[4]")
        if flags[1]:
            get_section_id(mock_cell, "TestCell[0].node[5]") == 4 * 10 + 5 + flags[0]*10
        else:
            with pytest.raises(SectionIdError):
                get_section_id(mock_cell, "TestCell[0].node[5]")
        if flags[2]:
            get_section_id(mock_cell, "TestCell[0].myelin[6]") == 4 * 10 + 6 + flags[0]*10 + flags[1]*10
        else:
            with pytest.raises(SectionIdError):
                get_section_id(mock_cell, "TestCell[0].myelin[6]")


def test_get_sec():
    mock_cell = _make_mock_cell()
    seal(mock_cell)
    assert get_sec(mock_cell, 9) == 9
    
    assert get_sec(mock_cell, 10) == 10
    assert get_sec(mock_cell, 11) == 11
    with pytest.raises(SectionIdError):
        get_sec(mock_cell, 12)
    assert get_sec(mock_cell, 20) == 12
    assert get_sec(mock_cell, 21) == 13
    with pytest.raises(SectionIdError):
        get_sec(mock_cell, 1111)