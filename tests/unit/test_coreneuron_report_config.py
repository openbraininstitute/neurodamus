from logging import config
import pytest
from neurodamus.core.coreneuron_report_config import CoreReportConfigEntry, CoreReportConfig
from neurodamus.report_parameters import ReportParameters, ReportType, Scaling, SectionType, Compartments

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
    with pytest.raises(ValueError):
        CoreReportConfigEntry(
            "r1", "target", "type", "var", "unit", "fmt",
            "sections", "compartments", "wrong", 0.0, 1.0,
            buffer_size=8, scaling="scale"
        )

def test_missing_required_fields():
    with pytest.raises(TypeError):
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
    assert entry.gids == [1, 2, 3]

def test_dump_load_mandatory_pop_offsets(tmp_path):
    file_path = tmp_path / "report.conf"

    config = CoreReportConfig()

    # --- Step 1: single regular entry ---
    e1 = CoreReportConfigEntry(
        "r1", "t1", "type1", "var1", "unit", "fmt",
        "sec1", "comp1", 0.1, 0.0, 1.0, buffer_size=8, scaling="scale"
    )
    e1.set_gids([1, 2, 3])
    config.add_entry(e1)

    # Set mandatory pop_offsets and optional spike filename
    config.set_pop_offsets({"pop1": 10, "pop2": 20})
    config.set_spike_filename("/path/to/spikes.dat")
    config.dump(file_path)

    loaded = CoreReportConfig.load(file_path)
    assert loaded == config
    assert loaded.pop_offsets == {"pop1": 10, "pop2": 20}
    assert loaded.spike_filename == "spikes.dat"

    # --- Step 2: add a compartment_set entry with new pop_offsets ---
    e2 = CoreReportConfigEntry(
        "r2", "t2", "compartment_set", "var2", "unit", "fmt",
        "sec2", "comp2", 0.2, 0.0, 2.0, buffer_size=16, scaling="scale"
    )
    e2.set_points([4,5], [10,20], [100,200])
    loaded.add_entry(e2)
    loaded.set_pop_offsets({"pop3": 30})  # must always set
    loaded.spike_filename = None           # optional
    loaded.dump(file_path)

    reloaded = CoreReportConfig.load(file_path)
    assert reloaded == loaded
    assert reloaded.pop_offsets == {"pop3": 30}
    assert reloaded.spike_filename is None

    # --- Step 3: add another compartment_set with spike filename ---
    e3 = CoreReportConfigEntry(
        "r3", "t3", "compartment_set", "var3", "unit", "fmt",
        "sec3", "comp3", 0.3, 0.0, 3.0, buffer_size=4, scaling="scale"
    )
    e3.set_points([6,7], [30,40], [300,400])
    reloaded.add_entry(e3)
    reloaded.set_pop_offsets({"pop4": 40})
    reloaded.set_spike_filename("/spikes2.dat")
    reloaded.dump(file_path)

    reloaded2 = CoreReportConfig.load(file_path)
    assert reloaded2 == reloaded
    assert reloaded2.pop_offsets == {"pop4": 40}
    assert reloaded2.spike_filename == "spikes2.dat"

    # --- Step 4: add normal entry without spike filename ---
    e4 = CoreReportConfigEntry(
        "r4", "t4", "type4", "var4", "unit", "fmt",
        "sec4", "comp4", 0.4, 0.0, 4.0, buffer_size=12, scaling="scale"
    )
    e4.set_gids([11,12,13])
    reloaded2.add_entry(e4)
    reloaded2.set_pop_offsets({"pop5": 50})  # mandatory
    reloaded2.spike_filename = None
    reloaded2.dump(file_path)

    reloaded3 = CoreReportConfig.load(file_path)
    assert reloaded3 == reloaded2
    assert reloaded3.pop_offsets == {"pop5": 50}
    assert reloaded3.spike_filename is None

    # --- Step 5: replace one compartment_set with a normal entry and set optional members ---
    replacement = CoreReportConfigEntry(
        "r3", "t3_new", "type_new", "var3_new", "unit", "fmt",
        "sec3_new", "comp3_new", 0.33, 0.0, 3.3, buffer_size=5, scaling="scale"
    )
    replacement.set_gids([7,8,9])
    reloaded3.add_entry(replacement)
    reloaded3.set_pop_offsets({"pop_new": 42})
    reloaded3.set_spike_filename("/new_spikes.dat")
    reloaded3.dump(file_path)

    final_loaded = CoreReportConfig.load(file_path)
    assert final_loaded == reloaded3
    assert final_loaded.pop_offsets == {"pop_new": 42}
    assert final_loaded.spike_filename == "new_spikes.dat"

def test_update_file(tmp_path):
    # Arrange: create a dummy config with one report
    file_path = tmp_path / "report.conf"
    conf = CoreReportConfig()
    e1 = CoreReportConfigEntry(
        "r1", "t1", "type1", "var1", "unit", "fmt",
        "sec1", "comp1", 0.1, 0.0, 1.0, buffer_size=8, scaling="scale"
    )
    e1.set_gids([1, 2, 3])
    conf.add_entry(e1)
    conf.set_pop_offsets({"pop1": 10, "pop2": 20})
    conf.set_spike_filename("/path/to/spikes.dat")
    conf.dump(file_path)

    # Act: update population_count
    substitutions = {"r1": {"buffer_size": 11}}
    CoreReportConfig.update_file(file_path, substitutions)

    # Assert: reloaded config has updated value
    reloaded = CoreReportConfig.load(file_path)
    assert reloaded.reports["r1"].buffer_size == 11

def test_update_file_failures(tmp_path):
    # Arrange
    file_path = tmp_path / "report.conf"
    conf = CoreReportConfig()
    e1 = CoreReportConfigEntry(
        "r1", "t1", "type1", "var1", "unit", "fmt",
        "sec1", "comp1", 0.1, 0.0, 1.0, buffer_size=8, scaling="scale"
    )
    e1.set_gids([1, 2, 3])
    conf.add_entry(e1)
    conf.set_pop_offsets({"pop1": 10})  # mandatory
    conf.dump(file_path)

    # --- Case 1: missing report key ---
    with pytest.raises(KeyError):
        CoreReportConfig.update_file(file_path, {"nonexistent": {"buffer_size": 11}})

    # --- Case 2: missing attribute on report ---
    with pytest.raises(AttributeError):
        CoreReportConfig.update_file(file_path, {"r1": {"nonexistent_attr": 42}})

    # --- Case 3: wrong type for existing attribute ---
    with pytest.raises(TypeError):
        CoreReportConfig.update_file(file_path, {"r1": {"buffer_size": "wrong_type"}})




