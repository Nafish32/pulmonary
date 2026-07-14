"""Cross-checkpoint benchmark harness (phase-5 comparison artifact).

Pure parsing/table-writing logic here (no torch) so it's CI-testable; the
model-bound driver that actually runs eval_from_weights per checkpoint lives in
``scripts/compare_checkpoints.py`` (Kaggle-only, needs torch/ultralytics/GPU).

``parse_results_md`` reads exactly the line formats ``pipeline._evaluate``
writes today (see src/pipeline.py). If that report format changes, this parser
must change with it -- there is no schema shared between the two, by design
(results.md is meant to be human-readable first).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

_PATTERNS: dict[str, re.Pattern] = {
    "detector": re.compile(r"^- detector loaded: `(.+)`", re.M),
    "map50": re.compile(r"^- \*\*mAP@50\*\*: ([\d.]+)", re.M),
    "ece": re.compile(r"^- ECE \(\d+ bins\): ([\d.]+)", re.M),
    "brier": re.compile(r"^- Brier: ([\d.]+)", re.M),
    "temperature": re.compile(r"^- temperature T: ([\d.]+)", re.M),
    "aurc": re.compile(r"^- AURC \(risk-coverage\): ([\d.]+)", re.M),
    "xai_energy": re.compile(r"^\s*- eigencam: ([\d.]+) \(n=(\d+)\)", re.M),
    "xai_baseline": re.compile(r"uniform baseline=([\d.]+)", re.M),
    "robustness_clean_map": re.compile(r"^- robustness .*clean mAP@50=([\d.]+)", re.M),
    "robustness_worst_map": re.compile(r"worst case .*mAP@50=([\d.]+)", re.M),
    "external_auroc": re.compile(r"^- external \(VinDr.*triage AUROC=([\d.]+)", re.M),
    "external_ece": re.compile(r"image-level ECE=([\d.]+)", re.M),
}

# lines that mean "stage ran but produced nothing usable" -- not a parse failure.
_SKIP_MARKERS = ("skipped", "FAILED", "n/a")


def parse_results_md(text: str) -> dict[str, float | str | None]:
    """Extract the fixed metric set from one ``results.md`` (or its text).

    Missing/skipped/failed stages come back as ``None``, never a guessed number.
    """
    out: dict[str, float | str | None] = {}
    det = _PATTERNS["detector"].search(text)
    out["detector"] = det.group(1) if det else None
    for key in ("map50", "ece", "brier", "temperature", "aurc",
                "robustness_clean_map", "robustness_worst_map",
                "external_auroc", "external_ece", "xai_baseline"):
        m = _PATTERNS[key].search(text)
        out[key] = float(m.group(1)) if m else None
    m = _PATTERNS["xai_energy"].search(text)
    out["xai_energy_in_box"] = float(m.group(1)) if m else None
    out["xai_n"] = int(m.group(2)) if m else None
    return out


def build_comparison_rows(entries: list[dict]) -> list[dict]:
    """``entries``: list of dicts with at least ``label`` + ``results_md_text``
    (+ optional ``param_count``, ``latency_ms_per_img``, ``train_hours`` compute
    -cost fields the driver measured itself -- never invented here)."""
    rows = []
    for e in entries:
        row = {"label": e["label"]}
        row.update(parse_results_md(e["results_md_text"]))
        for cost_key in ("param_count", "latency_ms_per_img", "train_hours"):
            row[cost_key] = e.get(cost_key)
        rows.append(row)
    return rows


def write_comparison_table(rows: list[dict], out_path: str) -> None:
    """Write the model x metric comparison table as CSV or Markdown, by extension."""
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].keys())
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        return
    # markdown table (default for any other extension, e.g. .md)
    lines = ["| " + " | ".join(fieldnames) + " |",
             "|" + "|".join("---" for _ in fieldnames) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("n/a" if row[k] is None else str(row[k])
                                        for k in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n")
