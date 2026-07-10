"""Typed, validated config schema.

Every CONFIG key is typed and checked at load time. Unknown keys (typos) hard-fail
via ``extra="forbid"`` -- this replaces the old brittle regex-patching of
``fast_mode`` and makes a typo fail immediately, not 40 minutes into training.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SplitConfig(BaseModel):
    """Patient-wise split fractions. Default 70/15/15 (held-out test != val)."""

    model_config = ConfigDict(extra="forbid")

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42

    @field_validator("train", "val", "test")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("split fraction must be in (0, 1)")
        return v

    @model_validator(mode="after")
    def _sum_to_one(self) -> "SplitConfig":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {total}")
        return self


class Config(BaseModel):
    """Full pipeline config. Unknown keys rejected -> typos fail at load."""

    model_config = ConfigDict(extra="forbid")

    # --- run identity ---
    seed: int = 42
    debug_mode: bool = False
    fast_mode: bool  # REQUIRED, no default -- a thesis run can't default into fast mode.

    # --- image / detector ---
    imgsz: int = 640
    fallback_imgsz: int = 512
    batch: int = 12
    epochs: int  # REQUIRED, no default.
    png_size: int = 640
    detector_model_name: str = "yolo26m.pt"
    detector_fallback_chain: list[str] = Field(
        default_factory=lambda: ["yolo26m.pt", "yolo11m.pt", "yolov8m.pt"]
    )
    optional_compare_model_name: str | None = None
    num_workers: int = 8
    cache_png: bool = True

    # --- stages ---
    calibration_enabled: bool = True
    uncertainty_enabled: bool = True
    xai_methods: list[str] = Field(
        default_factory=lambda: ["eigencam", "gradcam", "scorecam"]
    )
    robustness_enabled: bool = True
    n_bins: int = 15  # shared by ECE and reliability diagram (was mismatched 15 vs 10)

    # --- data / paths / reproducibility ---
    input_root: str = "/kaggle/input"
    working_root: str = "/kaggle/working"
    kaggle_dataset_version: str | None = None  # record in results.md alongside git SHA
    split: SplitConfig = Field(default_factory=SplitConfig)

    # --- experiment tracking (optional) ---
    wandb_enabled: bool = False

    @field_validator("epochs", "batch", "imgsz", "png_size")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v
