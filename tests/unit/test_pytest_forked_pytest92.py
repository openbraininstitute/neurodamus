"""Compatibility of pytest-forked report serialization with pytest >= 9.2."""

import marshal

import pytest
from _pytest.reports import TestReport

pytest.importorskip("pytest_forked")


def test_forked_serialize_report_is_marshallable():
    """pytest 9.2 NodeId in TestReport must not break pytest-forked marshal."""
    # tests/conftest.py::pytest_configure patches serialize_report for this.
    import pytest_forked

    rep = TestReport(
        "tests/unit/test_pytest_forked_pytest92.py::test_forked_serialize_report_is_marshallable",
        (__file__, 1, "test_forked_serialize_report_is_marshallable"),
        {},
        "passed",
        None,
        "call",
        [],
        0.1,
        0.0,
        0.1,
        [],
    )
    payload = pytest_forked.serialize_report(rep)
    assert "_id" not in payload
    rebuilt = TestReport(**marshal.loads(marshal.dumps(payload)))
    assert rebuilt.nodeid == rep.nodeid
    assert rebuilt.outcome == "passed"
