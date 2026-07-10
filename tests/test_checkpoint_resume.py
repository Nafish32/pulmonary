"""checkpoint_path: stable resume location survives repo re-clone.

No ultralytics/torch needed -- this only checks the path math + existence
branch, which is what decides fresh-train vs resume in train_detector.
"""

from __future__ import annotations

from src.models.train import checkpoint_path


class _Cfg:
    def __init__(self, working_root, fast_mode=False):
        self.working_root = str(working_root)
        self.fast_mode = fast_mode


def test_path_sits_outside_repo_clone(tmp_path):
    cfg = _Cfg(tmp_path)
    p = checkpoint_path(cfg)
    assert p == tmp_path / "outputs" / "runs" / "train_thesis" / "weights" / "last.pt"
    assert not p.exists()


def test_existing_checkpoint_detected(tmp_path):
    cfg = _Cfg(tmp_path)
    p = checkpoint_path(cfg)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"fake weights")
    assert checkpoint_path(cfg).exists()


def test_fast_and_thesis_use_separate_slots(tmp_path):
    # Switching configs in the same session must not resume the other's checkpoint.
    thesis_p = checkpoint_path(_Cfg(tmp_path, fast_mode=False))
    fast_p = checkpoint_path(_Cfg(tmp_path, fast_mode=True))
    assert thesis_p != fast_p

    fast_p.parent.mkdir(parents=True)
    fast_p.write_bytes(b"fake fast-probe weights")
    assert fast_p.exists()
    assert not thesis_p.exists()  # thesis run must NOT see the fast probe's checkpoint
