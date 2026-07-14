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
    # "" = ultralytics auto (single GPU 0). "0,1" = DDP across both T4/P100 (~2x
    # faster train, same effective batch). Multi-GPU DDP in a Kaggle notebook can
    # be flaky -> fall back to "0" if it errors; checkpoint-resume protects progress.
    device: str = ""
    detector_model_name: str = "yolo26m.pt"
    detector_fallback_chain: list[str] = Field(
        default_factory=lambda: ["yolo26m.pt", "yolo11m.pt", "yolov8m.pt"]
    )
    num_workers: int = 8
    cache_png: bool = True
    max_patients: int | None = None  # subset to N random patients (quick-but-real probe); None = all

    # --- stages ---
    calibration_enabled: bool = True
    uncertainty_enabled: bool = True
    xai_enabled: bool = True
    xai_samples: int = 20  # positive test images for saliency energy-in-box (EigenCAM)
    robustness_enabled: bool = True
    robustness_samples: int = 200  # test-image subset for the sweep (full set = 15x inference)
    # --- training-time corruption augmentation ---
    # Offline (export-time) copies, not an Ultralytics augmentation-pipeline hook:
    # hooking trainer-internal batch tensors is version-fragile and unverifiable
    # without a live torch/ultralytics install (see gotcha in CLAUDE.md). Instead,
    # export_split writes extra corrupted train images/labels (same corrupt()
    # used by the robustness eval sweep), bounded by train_corruption_frac. This
    # is the direct response to robustness_status: clean mAP@50 0.4563 collapsing
    # to 0.0057 under gaussian_noise severity 3 with zero training-time exposure
    # to any corruption. Severity 3 deliberately excluded from training (kept for
    # eval only) -- training on the most extreme corruption risks hurting clean
    # accuracy more than it buys robustness; 1-2 is the common practice middle
    # ground. NOT VERIFIED end-to-end (no GPU train run yet) -- check the
    # resulting train/mAP@50 clean vs robustness-sweep numbers on the next
    # Kaggle run.
    train_corruption_aug_enabled: bool = True
    train_corruption_kinds: list[str] = Field(
        default_factory=lambda: ["gaussian_noise", "blur", "contrast"]
    )
    train_corruption_severities: list[int] = Field(default_factory=lambda: [1, 2])
    train_corruption_frac: float = 0.15  # fraction of train images duplicated, corrupted
    external_enabled: bool = True  # VinDr cross-domain eval; skips cleanly if VinDr absent
    # >1 seed => train that many members and report ensemble spread. Each extra seed
    # is a FULL train (~12hr on thesis.yaml), so default is single-model (no spread).
    ensemble_seeds: list[int] = Field(default_factory=lambda: [42])
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

    @field_validator("train_corruption_kinds")
    @classmethod
    def _valid_corruption_kinds(cls, v: list[str]) -> list[str]:
        from src.robustness.corruption import CORRUPTIONS

        bad = set(v) - set(CORRUPTIONS)
        if bad:
            raise ValueError(f"unknown corruption kind(s) {bad}; pick from {CORRUPTIONS}")
        return v

    @field_validator("train_corruption_severities")
    @classmethod
    def _valid_corruption_severities(cls, v: list[int]) -> list[int]:
        bad = [s for s in v if not 1 <= s <= 3]
        if bad:
            raise ValueError(f"severities must be 1-3, got {bad}")
        return v

    @field_validator("train_corruption_frac")
    @classmethod
    def _frac_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("train_corruption_frac must be in [0, 1]")
        return v
