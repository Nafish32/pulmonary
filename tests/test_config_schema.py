"""Config schema fails fast on typos, missing required fields, bad splits."""

import pytest
from pydantic import ValidationError

from src.config.schema import Config, SplitConfig


def test_unknown_key_rejected():
    # A typo'd key must hard-fail at load, not be silently ignored.
    with pytest.raises(ValidationError):
        Config(fast_mode=False, epochs=40, fastmode=True)


def test_fast_mode_required():
    with pytest.raises(ValidationError):
        Config(epochs=40)  # no fast_mode


def test_epochs_required():
    with pytest.raises(ValidationError):
        Config(fast_mode=False)  # no epochs


def test_epochs_must_be_positive():
    with pytest.raises(ValidationError):
        Config(fast_mode=False, epochs=0)


def test_splits_must_sum_to_one():
    with pytest.raises(ValidationError):
        SplitConfig(train=0.7, val=0.7, test=0.15)


def test_valid_config_defaults():
    c = Config(fast_mode=False, epochs=40)
    assert c.split.train + c.split.val + c.split.test == pytest.approx(1.0)
    assert c.n_bins == 15
    assert c.detector_fallback_chain[0] == "yolo26m.pt"
