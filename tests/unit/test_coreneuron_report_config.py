import pytest
from neurodamus.core.coreneuron_report_config import CoreReportConfigEntry

def test_init_positional_and_kwargs():
    # positional only
    entry1 = CoreReportConfigEntry(
        "r1", "target", "type", "var", "unit", "fmt", "t_type", 0.1, 0.0, 1.0, buffer_size=8
    )
    # kwargs only
    entry2 = CoreReportConfigEntry(
        report_name="r2", target_name="tgt", report_type="t", report_variable="v",
        unit="u", report_format="f", target_type="tt", dt=0.2, start_time=0.0,
        end_time=2.0, buffer_size=16
    )
    assert entry1.buffer_size == 8
    assert entry2.dt == 0.2

def test_type_enforcement():
    with pytest.raises(AssertionError):
        CoreReportConfigEntry(
            "r1", "target", "type", "var", "unit", "fmt", "t_type", "wrong", 0.0, 1.0, buffer_size=8
        )

def test_missing_required_fields():
    with pytest.raises(ValueError):
        CoreReportConfigEntry(
            "r1", "target", "type", "var", "unit", "fmt", "t_type", 0.1, 0.0, 1.0
        )

def test_set_gids_and_num_gids():
    entry = CoreReportConfigEntry(
        "r1", "target", "type", "var", "unit", "fmt", "t_type", 0.1, 0.0, 1.0, buffer_size=8
    )
    assert entry.num_gids == 0
    entry.set_gids([1, 2, 3])
    assert entry.num_gids == 3
    assert entry.gids == [1, 2, 3]

def test_equality_and_repr():
    e1 = CoreReportConfigEntry(
        "r1", "target", "type", "var", "unit", "fmt", "t_type", 0.1, 0.0, 1.0, buffer_size=8
    )
    e2 = CoreReportConfigEntry(
        "r1", "target", "type", "var", "unit", "fmt", "t_type", 0.1, 0.0, 1.0, buffer_size=8
    )
    e1.set_gids([1,2,3])
    e2.set_gids([1,2,3])
    assert e1 == e2
    e2.set_gids([4,5])
    assert e1 != e2
    # repr returns string containing field names
    r = repr(e1)
    for name, _, _ in e1.SLOTS:
        assert name in r
    assert "gids" in r
