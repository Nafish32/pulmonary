"""Group-wise (patient) splits and leakage assertions.

Patient leakage across train/val/test silently inflates every internal number,
so the assertion lives here and is unit-tested independently of the split logic.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def assert_no_leakage(
    train_ids: Iterable,
    val_ids: Iterable,
    test_ids: Iterable | None = None,
) -> None:
    """Assert no patient id appears in more than one split.

    Raises:
        AssertionError: With the overlapping split pair, count, and 3 sample ids.
    """
    train, val = set(train_ids), set(val_ids)
    pairs = [("train", "val", train & val)]
    if test_ids is not None:
        test = set(test_ids)
        pairs += [("train", "test", train & test), ("val", "test", val & test)]

    leaks = [(a, b, s) for a, b, s in pairs if s]
    if leaks:
        detail = "; ".join(
            f"{a}∩{b}={len(s)} e.g. {list(s)[:3]}" for a, b, s in leaks
        )
        raise AssertionError(f"patient leakage detected: {detail}")


def patient_split(df, split, id_col: str = "patientId"):
    """Split ``df`` patient-wise into train/val/test frames.

    Args:
        df: Label dataframe with an id column.
        split: A ``SplitConfig`` (train/val/test fractions + seed).
        id_col: Column holding the patient/group id.

    Returns:
        (train_df, val_df, test_df). Assert-checked for zero patient leakage.
    """
    ids = np.asarray(df[id_col].unique())  # numpy, not Arrow -> shuffle is safe
    rng = np.random.default_rng(split.seed)
    rng.shuffle(ids)  # in place, seeded -> reproducible

    n = len(ids)
    n_train = int(round(split.train * n))
    n_val = int(round(split.val * n))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    assert_no_leakage(train_ids, val_ids, test_ids)
    pick = lambda s: df[df[id_col].isin(s)].copy()  # noqa: E731
    return pick(train_ids), pick(val_ids), pick(test_ids)
