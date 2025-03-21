"""
ProgressBar test suite
"""

import pytest
from neurodamus.utils.progressbar import ProgressBar


@pytest.mark.parametrize("progress_value, expected_progress", [
    (20, 20),
    (40, 40),
    (60, 60),
    (80, 80),
    (100, 100)
])
def test_progressbar_increment(progress_value, expected_progress):
    """Test the incremental progress of the ProgressBar."""
    p = ProgressBar(100, width=80)
    # Simulate incrementing the progress by 20 each time
    while p.progress < progress_value:
        p += 20

    assert p.progress == expected_progress, ""
    f"Expected progress {expected_progress}, but got {p.progress}"


def test_progressbar_decrement():
    """Test the decrementing progress of the ProgressBar."""
    p = ProgressBar(100, width=80)

    # Set progress to a value and decrement
    for i in range(80, -1, -20):
        p.progress = i

    assert p.progress == 0, f"Expected progress 0, but got {p.progress}"


def test_progressbar_consuming_generator():
    """Test if ProgressBar can be used as a consumer-generator."""
    l1 = range(0, 100, 10)

    # Iterate over the generator and check if ProgressBar is consuming as expected
    for _ in ProgressBar.iter(l1):
        pass  # Simulating work done with each iteration

    # Since the test doesn't need assertions, we just check no errors happen


def test_progressbar_with_sub_selection():
    """Test the reuse of a ProgressBar instance with sub-selection of a range."""
    bar = ProgressBar(15)
    l2 = range(100, 200, 10)

    # Progress bar will consume the first range
    for _ in bar(l2):
        pass

    # Apply sub-selection
    l2 = list(bar(l2, 5))

    # Assert that the sub-selection produces the expected result
    expected = [100, 110, 120, 130, 140]
    assert l2 == expected, f"Expected {expected}, but got {l2}"


def test_progressbar_with_spinner():
    """Test ProgressBar with False to create a spinner."""
    bar = ProgressBar(False)

    # Iterate over the range with a spinner enabled
    for _ in bar(range(60)):
        pass

    # No assertion is needed since we're testing the spinner functionality
    # Just ensure no errors occur


@pytest.mark.parametrize("start, expected_value", [
    (100, 100),
    (50, 50),
    (0, 0)
])
def test_progressbar_initial_value(start, expected_value):
    """Test the ProgressBar with custom initial values."""
    p = ProgressBar(100, width=80, start=start)
    assert p.progress == expected_value, ""
    f"Expected initial value {expected_value}, but got {p.progress}"
