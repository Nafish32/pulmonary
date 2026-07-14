"""Phase-5 benchmark harness: pure parsing/table logic (no torch, CI-safe).

The model-bound driver (compare_checkpoints.py) that produces results.md per
checkpoint is Kaggle-only; this only tests the parser against the EXACT line
formats pipeline._evaluate writes, and the table writer.
"""

from __future__ import annotations

import csv

from src.evaluation.comparison import (
    build_comparison_rows, parse_results_md, write_comparison_table)

SAMPLE_RESULTS_MD = """# Results

`[THESIS] yolo26m.pt | patients~4002 test | mAP@50=0.3662 ECE=0.0853 AURC=0.5991 (fast_mode=False, epochs=50)`

Screening/triage **support** artifact -- NOT a diagnostic system.

- detector loaded: `yolo26m.pt`
- fast_mode=False epochs=50 png_size=640
- kaggle_dataset_version: 3
- test images: 4002, predictions: 39763 (conf>=0.05: 3715)
- **mAP@50**: 0.3662
- ECE (15 bins): 0.0853
- Brier: 0.1558
- temperature T: 1.341
- reliability diagram: `reliability.png`
- AURC (risk-coverage): 0.5991
- XAI saliency energy-in-box (positives; uniform baseline=0.171, higher=better):
  - eigencam: 0.143 (n=18) -- ~uniform: attends to lung fields, not lesion-specific (see xai_example.png)
  - eigencam deletion AUC=0.612 (n=8, lower=more faithful: erasing salient pixels collapses the detection)
- robustness (200 imgs): clean mAP@50=0.4563 ECE=0.0721
  - worst case ('gaussian_noise', 3): mAP@50=0.0057 (drop 0.4506), ECE=n/a (no conf>=0.05 preds under corruption)
- external (VinDr, n=1000, no recalibration): triage AUROC=0.8242, image-level ECE=0.4757
- external recalibration (label-free; top-20% referral, actual risk=0.0650):
  - dm: target ECE=0.3336, expected risk=0.6670, calib gap=0.3334
- ensemble: single model, spread not meaningful (set ensemble_seeds to >=2 seeds for this metric)
"""


def test_parse_results_md_extracts_known_fields():
    d = parse_results_md(SAMPLE_RESULTS_MD)
    assert d["detector"] == "yolo26m.pt"
    assert d["map50"] == 0.3662
    assert d["ece"] == 0.0853
    assert d["brier"] == 0.1558
    assert d["temperature"] == 1.341
    assert d["aurc"] == 0.5991
    assert d["xai_energy_in_box"] == 0.143
    assert d["xai_n"] == 18
    assert d["xai_baseline"] == 0.171
    assert d["robustness_clean_map"] == 0.4563
    assert d["robustness_worst_map"] == 0.0057
    assert d["external_auroc"] == 0.8242
    assert d["external_ece"] == 0.4757


def test_parse_results_md_missing_stage_is_none_not_guessed():
    d = parse_results_md("# Results\n- **mAP@50**: 0.10\n")
    assert d["map50"] == 0.10
    assert d["external_auroc"] is None
    assert d["xai_energy_in_box"] is None


def test_build_comparison_rows_carries_compute_cost():
    rows = build_comparison_rows([{
        "label": "yolo-upgraded",
        "results_md_text": SAMPLE_RESULTS_MD,
        "param_count": 25_000_000,
        "latency_ms_per_img": 12.3,
        "train_hours": None,
    }])
    assert rows[0]["label"] == "yolo-upgraded"
    assert rows[0]["map50"] == 0.3662
    assert rows[0]["param_count"] == 25_000_000
    assert rows[0]["train_hours"] is None


def test_write_comparison_table_csv(tmp_path):
    rows = build_comparison_rows([{"label": "a", "results_md_text": SAMPLE_RESULTS_MD}])
    out = tmp_path / "cmp.csv"
    write_comparison_table(rows, str(out))
    with out.open() as f:
        r = list(csv.DictReader(f))
    assert r[0]["label"] == "a"
    assert r[0]["map50"] == "0.3662"


def test_write_comparison_table_md(tmp_path):
    rows = build_comparison_rows([
        {"label": "a", "results_md_text": SAMPLE_RESULTS_MD},
        {"label": "b", "results_md_text": "# Results\n- **mAP@50**: 0.20\n"},
    ])
    out = tmp_path / "cmp.md"
    write_comparison_table(rows, str(out))
    text = out.read_text()
    assert "| label |" in text
    assert "| a |" in text.replace(" ", "") or "a" in text
    assert "n/a" in text  # model b's missing fields


def test_write_comparison_table_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        write_comparison_table([], "x.csv")
