from unittest.mock import Mock, seal
import pytest

from neurodamus.metype import BaseCell, SectionIdError

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
    mock_cell = _make_mock_cell()
    seal(mock_cell)
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].soma[0]") == 0 * 10
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].axon[1]") == 1 * 10 + 1
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].dend[2]") == 2 * 10 + 2
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].apic[3]") == 3 * 10 + 3
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].ais[4]") == 4 * 10 + 4
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].node[5]") == 4 * 10 + 5
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].myelin[6]") == 4 * 10 + 6
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].myelin[6]") == 4 * 10 + 6

def test_get_sec():
    mock_cell = _make_mock_cell()
    seal(mock_cell)
    assert BaseCell.get_sec(mock_cell, 9) == 9
    
    assert BaseCell.get_sec(mock_cell, 10) == 10
    assert BaseCell.get_sec(mock_cell, 11) == 11
    with pytest.raises(SectionIdError):
        BaseCell.get_sec(mock_cell, 12)
    assert BaseCell.get_sec(mock_cell, 20) == 12
    assert BaseCell.get_sec(mock_cell, 21) == 13
    with pytest.raises(SectionIdError):
        BaseCell.get_sec(mock_cell, 1111)