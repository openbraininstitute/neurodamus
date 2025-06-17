from unittest.mock import Mock, seal

from neurodamus.metype import BaseCell

def _make_mock_cell():
    mock_cell = Mock()
    for sec_type in BaseCell._sections:
        setattr(mock_cell, sec_type, 10*[None])
    return mock_cell


# ["soma", "axon", "dend", "apic", "ais", "node", "myelin"]

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

# TODO test_get_sec