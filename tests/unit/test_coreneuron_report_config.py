import pytest
from neurodamus.core.coreneuron_report_config import CoreReportConfigEntry, CoreReportConfig

def test_init_positional_and_kwargs():
    # positional only
    entry1 = CoreReportConfigEntry(
        "r1", "target", "type", "var", "unit", "fmt",
        "sections", "compartments", 0.1, 0.0, 1.0, buffer_size=8, scaling="scale"
    )
    # kwargs only
    entry2 = CoreReportConfigEntry(
        report_name="r2", target_name="tgt", report_type="t", report_variable="v",
        unit="u", report_format="f", sections="sec", compartments="comp",
        dt=0.2, start_time=0.0, end_time=2.0, buffer_size=16, scaling="scale"
    )
    assert entry1.buffer_size == 8
    assert entry2.dt == 0.2

def test_type_enforcement():
    with pytest.raises(AssertionError):
        CoreReportConfigEntry(
            "r1", "target", "type", "var", "unit", "fmt",
            "sections", "compartments", "wrong", 0.0, 1.0,
            buffer_size=8, scaling="scale"
        )

def test_missing_required_fields():
    with pytest.raises(ValueError):
        CoreReportConfigEntry(
            "r1", "target", "type", "var", "unit", "fmt",
            "sections", "compartments", 0.1, 0.0, 1.0
        )

def test_set_gids_and_num_gids():
    entry = CoreReportConfigEntry(
        "r1", "target", "type", "var", "unit", "fmt",
        "sections", "compartments", 0.1, 0.0, 1.0, buffer_size=8, scaling="scale"
    )
    assert entry.num_gids == 0
    entry.set_gids([1, 2, 3])
    assert entry.num_gids == 3
    assert entry._gids == [1, 2, 3]

def test_equality_and_repr():
    e1 = CoreReportConfigEntry(
        "r1", "target", "type", "var", "unit", "fmt",
        "sections", "compartments", 0.1, 0.0, 1.0, buffer_size=8, scaling="scale"
    )
    e2 = CoreReportConfigEntry(
        "r1", "target", "type", "var", "unit", "fmt",
        "sections", "compartments", 0.1, 0.0, 1.0, buffer_size=8, scaling="scale"
    )
    e1.set_gids([1,2,3])
    e2.set_gids([1,2,3])
    assert e1 == e2
    e2.set_gids([4,5])
    assert e1 != e2
    # repr contains SLOTS names
    r = repr(e1)
    for name, _, _ in e1.SLOTS:
        assert name in r
    assert "gids" in r

def test_dump_load_modify_sequence(tmp_path):
    file_path = tmp_path / "report.conf"

    config = CoreReportConfig()

    # --- Step 1: single regular entry ---
    e1 = CoreReportConfigEntry(
        "r1", "t1", "type1", "var1", "unit", "fmt",
        "sec1", "comp1", 0.1, 0.0, 1.0, buffer_size=8, scaling="scale"
    )
    e1.set_gids([1, 2, 3])
    config.add_entry(e1)
    config.dump(file_path)

    loaded = CoreReportConfig.load(file_path)
    assert loaded == config

    # --- Step 2: add first compartment_set ---
    e2 = CoreReportConfigEntry(
        "r2", "t2", "compartment_set", "var2", "unit", "fmt",
        "sec2", "comp2", 0.2, 0.0, 2.0, buffer_size=16, scaling="scale"
    )
    e2.set_points([4,5], [10,20], [100,200])
    loaded.add_entry(e2)
    loaded.dump(file_path)

    reloaded = CoreReportConfig.load(file_path)
    assert reloaded == loaded

    # --- Step 3: add second compartment_set ---
    e3 = CoreReportConfigEntry(
        "r3", "t3", "compartment_set", "var3", "unit", "fmt",
        "sec3", "comp3", 0.3, 0.0, 3.0, buffer_size=4, scaling="scale"
    )
    e3.set_points([6,7], [30,40], [300,400])
    reloaded.add_entry(e3)
    reloaded.dump(file_path)

    reloaded2 = CoreReportConfig.load(file_path)
    assert reloaded2 == reloaded

    # --- Step 4: add a normal entry ---
    e4 = CoreReportConfigEntry(
        "r4", "t4", "type4", "var4", "unit", "fmt",
        "sec4", "comp4", 0.4, 0.0, 4.0, buffer_size=12, scaling="scale"
    )
    e4.set_gids([11,12,13])
    reloaded2.add_entry(e4)
    reloaded2.dump(file_path)

    reloaded3 = CoreReportConfig.load(file_path)
    assert reloaded3 == reloaded2

    # --- Step 5: replace one compartment_set with a normal entry ---
    replacement = CoreReportConfigEntry(
        "r3", "t3_new", "type_new", "var3_new", "unit", "fmt",
        "sec3_new", "comp3_new", 0.33, 0.0, 3.3, buffer_size=5, scaling="scale"
    )
    replacement.set_gids([7,8,9])
    reloaded3.add_entry(replacement)
    reloaded3.dump(file_path)

    final_loaded = CoreReportConfig.load(file_path)
    assert final_loaded == reloaded3

