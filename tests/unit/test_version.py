# test_git_version.py

import pytest
from neurodamus.utils.version import GitVersion

def test_parsing():
    v = GitVersion("1.2.3-45-gabcdef")
    assert str(v.base) == "1.2.3"
    assert v.commits == 45
    assert v.commit == "abcdef"

    v2 = GitVersion("1.2.3")
    assert str(v2.base) == "1.2.3"
    assert v2.commits == 0
    assert v2.commit == ""

def test_comparison():
    v1 = GitVersion("1.2.3-10-gaaaaaa")
    v2 = GitVersion("1.2.3-15-gbbbbbb")
    v3 = GitVersion("1.2.4-5-gcccccc")
    v4 = GitVersion("1.2.3-10-gaaaaaa")
    v5 = GitVersion("9.0a")
    v6 = GitVersion("9.0.0")
    v7 = GitVersion("9.0")

    assert v1 < v2
    assert v2 < v3
    assert v1 == v4
    assert v3 > v2
    assert v5 < v6
    assert v5 < v7
    assert v6 == v7
