"""Synthetic end-to-end smoke test.

Skeleton state: verifies the pipeline entrypoint is importable and callable. The
full 5-fake-image run (discover -> cache -> export -> 1-epoch train -> ... ->
artifact save on CPU, <2 min) is skipped until stages are implemented. Un-skip
it stage by stage as src/pipeline.run_all gets wired.
"""

import importlib.util

import pytest

_HAS_STACK = importlib.util.find_spec("ultralytics") is not None


def test_pipeline_entrypoint_importable():
    from src.pipeline import run_all

    assert callable(run_all)


def test_config_loads_from_debug_yaml():
    from pathlib import Path

    from src.config.loader import load_config

    debug = Path(__file__).resolve().parents[1] / "configs" / "debug.yaml"
    cfg = load_config(debug)
    assert cfg.fast_mode is True
    assert cfg.debug_mode is True


@pytest.mark.skipif(
    not _HAS_STACK,
    reason="full run needs ultralytics/torch/pydicom/cv2 + GPU -- runs on Kaggle, not CI",
)
def test_full_pipeline_on_synthetic_5_images(tmp_path):
    # Stages are wired; on a box with the imaging+torch stack this exercises
    # discover -> cache -> export -> 1-epoch train -> eval -> results.md.
    import numpy as np
    import pandas as pd
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    from src.config.loader import load_config

    # 5 synthetic DICOMs + RSNA-shaped labels under a fake input_root.
    root = tmp_path / "input" / "rsna"
    imgs = root / "stage_2_train_images"
    imgs.mkdir(parents=True)
    rows = []
    for i in range(5):
        arr = (np.random.default_rng(i).integers(0, 255, (64, 64))).astype(np.uint16)
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.Rows, ds.Columns = 64, 64
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.PixelData = arr.tobytes()
        pid = f"p{i}"
        ds.save_as(imgs / f"{pid}.dcm", enforce_file_format=True)
        rows.append(dict(patientId=pid, x=10, y=10, width=20, height=20, Target=1))
    (root / "stage_2_train_labels.csv").write_text(pd.DataFrame(rows).to_csv(index=False))

    from pathlib import Path

    from src.pipeline import run_all

    cfg = load_config("configs/debug.yaml")
    object.__setattr__(cfg, "input_root", str(tmp_path / "input"))
    object.__setattr__(cfg, "working_root", str(tmp_path / "work"))
    results = run_all(cfg)
    assert Path(results).exists()
