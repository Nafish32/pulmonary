"""checkpoint_path: stable resume location survives repo re-clone.

No ultralytics/torch needed -- this only checks the path math + existence
branch, which is what decides fresh-train vs resume in train_detector.
"""

from __future__ import annotations

from src.models.train import checkpoint_path


class _Cfg:
    def __init__(self, working_root):
        self.working_root = str(working_root)


def test_path_sits_outside_repo_clone(tmp_path):
    cfg = _Cfg(tmp_path)
    p = checkpoint_path(cfg)
    assert p == tmp_path / "outputs" / "runs" / "train" / "weights" / "last.pt"
    assert not p.exists()


def test_existing_checkpoint_detected(tmp_path):
    cfg = _Cfg(tmp_path)
    p = checkpoint_path(cfg)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"fake weights")
    assert checkpoint_path(cfg).exists()
