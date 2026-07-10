"""Patient-leakage assertion catches overlap across splits."""

import pytest

from src.data.split import assert_no_leakage


def test_clean_three_way_split_ok():
    assert_no_leakage([1, 2, 3], [4, 5], [6, 7])


def test_train_val_overlap_raises():
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage([1, 2, 3], [3, 4])


def test_train_test_overlap_raises():
    with pytest.raises(AssertionError):
        assert_no_leakage([1, 2, 3], [4, 5], [3])


def test_error_names_sample_ids():
    with pytest.raises(AssertionError, match="e.g."):
        assert_no_leakage([1, 2], [2])
