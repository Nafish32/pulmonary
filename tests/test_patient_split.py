"""patient_split: patient-wise, leakage-free, reproducible."""

from __future__ import annotations

import pandas as pd

from src.config.schema import SplitConfig
from src.data.split import patient_split


def _multi_row_df(n_patients=100, rows_each=3):
    rows = []
    for p in range(n_patients):
        for _ in range(rows_each):
            rows.append(dict(patientId=f"p{p}", box=1))
    return pd.DataFrame(rows)


def test_no_patient_in_two_splits():
    df = _multi_row_df()
    tr, va, te = patient_split(df, SplitConfig())
    s_tr = set(tr.patientId)
    s_va = set(va.patientId)
    s_te = set(te.patientId)
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te)
    assert len(tr) + len(va) + len(te) == len(df)  # no rows dropped


def test_seed_reproducible():
    df = _multi_row_df()
    a = patient_split(df, SplitConfig(seed=7))[0]
    b = patient_split(df, SplitConfig(seed=7))[0]
    assert set(a.patientId) == set(b.patientId)
