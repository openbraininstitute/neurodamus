from unittest.mock import Mock, seal
import pytest

from neurodamus.metype import BaseCell

def _make_mock_cell():
    mock_cell = Mock()
    offset = 0
    for sec_type in BaseCell._sections:
        setattr(mock_cell, sec_type, [offset + i for i in range(10)])
        offset += 10
    mock_cell.unexpected_section = list(range(5))
    return mock_cell

def test_get_section_id():
    mock_cell = _make_mock_cell()
    seal(mock_cell)
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].soma[0]") == 0 * 10
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].axon[1]") == 1 * 10 + 1
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].dend[2]") == 2 * 10 + 2
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].apic[3]") == 3 * 10 + 3
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].ais[4]") == 4 * 10 + 4
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].node[5]") == 5 * 10 + 5
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].myelin[6]") == 6 * 10 + 6
    assert BaseCell.get_section_id(mock_cell, "TestCell[0].myelin[6]") == 6 * 10 + 6
    with pytest.raises(IndexError):
        BaseCell.get_section_id(mock_cell, "TestCell[0].myelin[11]")
    with pytest.raises(IndexError):
        BaseCell.get_section_id(mock_cell, "TestCell[0].myelin[-1]")
    with pytest.raises(ValueError):
        BaseCell.get_section_id(mock_cell, "TestCell[0].wrong_section[1]")
    with pytest.raises(RuntimeError):
        BaseCell.get_section_id(mock_cell, "TestCell[0].unexpected_section[1]")

def test_get_sec():
    mock_cell = _make_mock_cell()
    seal(mock_cell)
    assert BaseCell.get_sec(mock_cell, 9) == 9
    assert BaseCell.get_sec(mock_cell, 30) == 30
    assert BaseCell.get_sec(mock_cell, -1) == 7 * 10 - 1
    with pytest.raises(IndexError):
        BaseCell.get_sec(mock_cell, 70)
    with pytest.raises(IndexError):
        BaseCell.get_sec(mock_cell, -80)