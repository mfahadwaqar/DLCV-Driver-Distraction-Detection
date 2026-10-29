import numpy as np
import pytest

from src.demo.run import Classifier, Persistence, eye_bounds


def test_persistence_requires_continuity_and_missing_resets():
    gate = Persistence(seconds=1, max_gap=0.6)
    assert gate.update(0, 1) == []
    assert gate.update(0.5, 1) == []
    assert gate.update(1, 1) == [{"event": "start", "label": 1, "timestamp": 1}]
    assert gate.update(1.1, None) == [{"event": "end", "label": 1, "timestamp": 1.1}]
    assert gate.update(1.2, 1) == []
    assert gate.update(2.2, 1) == []  # A long observation gap restarts persistence.
    assert gate.update(2.7, 1) == []
    assert gate.update(3.2, 1)[0]["event"] == "start"
    with pytest.raises(ValueError, match="increase"):
        gate.update(3.2, 1)


def test_eye_crop_is_clamped_and_degenerate_crop_unknown():
    bounds = eye_bounds([[0, 2], [2, 1], [4, 1], [10, 2], [4, 3], [2, 3]], 12, 5)
    x1, y1, x2, y2 = bounds
    assert 0 <= x1 < x2 <= 12 and 0 <= y1 < y2 <= 5
    assert eye_bounds(np.zeros((6, 2)), 100, 100) is None
    assert eye_bounds(np.full((6, 2), np.nan), 100, 100) is None


def test_checkpoint_task_mismatch_fails_before_loading_weights(tmp_path):
    (tmp_path / "config.yaml").write_text("label_key: label_distraction\nclass_ids: [0, 1]\n")
    with pytest.raises(ValueError, match="Wrong checkpoint task"):
        Classifier(tmp_path, "label_drowsiness")
