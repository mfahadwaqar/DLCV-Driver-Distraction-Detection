import ast
import json
from pathlib import Path

import pytest


def resolver():
    notebook = Path(__file__).parents[1] / "notebooks/driver_monitoring_training.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    tree = ast.parse("".join(cells[4]["source"]))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "find_kaggle_source"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "discovery", "exec"), namespace)
    return namespace["find_kaggle_source"]


def snapshot(root):
    for name in [
        "src/train/phase1.py",
        "src/train/notebook_backup.py",
        "configs/state_farm_efficientnet.yaml",
    ]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")


@pytest.mark.parametrize(
    "layout",
    ["dlcv-project-code/project", "datasets/owner/dlcv-project-code/dlcv-project-code-v3/project"],
)
def test_source_found_with_both_mount_layouts_and_wrapper(tmp_path, layout):
    expected = tmp_path / layout
    snapshot(expected)
    assert resolver()(tmp_path) == expected.resolve()


def test_old_snapshot_is_not_selected_and_ambiguity_is_rejected(tmp_path):
    old = tmp_path / "dlcv-project-code/v2/project/src/train"
    old.mkdir(parents=True)
    (old / "phase1.py").write_text("old")
    assert resolver()(tmp_path) is None
    snapshot(tmp_path / "dlcv-project-code/v3/project")
    snapshot(tmp_path / "dlcv-project-code/v4/project")
    with pytest.raises(ValueError, match="Multiple EfficientNet source snapshots"):
        resolver()(tmp_path)
