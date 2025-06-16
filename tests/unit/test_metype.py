from unittest.mock import Mock, seal

from neurodamus.metype import BaseCell

def _make_mock_cell():
    mock_cell = Mock()
    mock_cell.soma = 10*[None]
    mock_cell.axon = 10*[None]
    mock_cell.dend = 10*[None]
    mock_cell.apic = 10*[None]
    return mock_cell


# ["soma", "axon", "dend", "apic", "ais", "node", "myelin"]

def test_get_section_index():
    mock_cell = _make_mock_cell()
    seal(mock_cell)
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].soma[0]") == 0 * 10
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].axon[1]") == 1 * 10 + 1
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].dend[2]") == 2 * 10 + 2
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].apic[3]") == 3 * 10 + 3
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].ais[4]") == 4 * 10 + 4
    # node / myelin depend on nSecLastAIS / nSecNodal which may not exist
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].node[5]") == 4 * 10 + 5
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].myelin[6]") == 4 * 10 + 6

    mock_cell = _make_mock_cell()
    mock_cell.nSecLastAIS = 10
    seal(mock_cell)
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].node[7]") == 5 * 10 + 7
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].myelin[8]") == 5 * 10 + 8

    mock_cell = _make_mock_cell()
    mock_cell.nSecLastAIS = 10
    mock_cell.nSecNodal = 10
    seal(mock_cell)
    assert BaseCell.get_section_index(mock_cell, "TestCell[0].myelin[9]") == 6 * 10 + 9

# TODO test_get_sec