"""Regression checks for the notebook's remote-file validation bottleneck."""

import ast
import json
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath
from time import perf_counter

import pytest


@pytest.fixture
def check_images():
    notebook = Path(__file__).parents[1] / "notebooks/driver_monitoring_training.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    tree = ast.parse("".join(cells[6]["source"]))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_prepared_images"
    )
    namespace = {
        "Path": Path,
        "PurePosixPath": PurePosixPath,
        "os": os,
        "defaultdict": defaultdict,
        "perf_counter": perf_counter,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "preflight", "exec"), namespace)
    return namespace["check_prepared_images"]


def test_files_are_checked_with_directory_bounded_resolves(
    check_images, tmp_path, monkeypatch, capsys
):
    rows = []
    for directory in ("a", "b"):
        (tmp_path / directory).mkdir()
        for i in range(50):
            relative = f"{directory}/{i}.jpg"
            (tmp_path / relative).write_bytes(b"path-check-only")
            rows.append({"path": relative})
    original = Path.resolve
    calls = []

    def counted(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted)
    check_images(tmp_path, rows, "sample")
    assert len(calls) == 3  # Root plus two parent directories, rather than 100 images.
    assert "100/100 paths checked" in capsys.readouterr().out


def test_directory_is_not_accepted_as_an_image(check_images, tmp_path):
    (tmp_path / "not-a-file.jpg").mkdir()
    with pytest.raises(FileNotFoundError, match="1 images missing"):
        check_images(tmp_path, [{"path": "not-a-file.jpg"}], "sample")


@pytest.mark.parametrize(
    "relative", ["../escape.jpg", "/absolute.jpg", "C:/outside.jpg", "a\\b.jpg"]
)
def test_path_escape_still_rejected(check_images, tmp_path, relative):
    with pytest.raises(ValueError, match="Invalid relative image path"):
        check_images(tmp_path, [{"path": relative}], "sample")


@pytest.mark.parametrize("directory_link", [False, True])
def test_symlink_escape_still_rejected(check_images, tmp_path, directory_link):
    root = tmp_path / "prepared"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "eye.jpg").write_bytes(b"outside")
    link = root / ("folder" if directory_link else "eye.jpg")
    try:
        link.symlink_to(
            outside if directory_link else outside / "eye.jpg", target_is_directory=directory_link
        )
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this host")
    with pytest.raises(ValueError, match="escapes prepared root"):
        check_images(root, [{"path": "folder/eye.jpg" if directory_link else "eye.jpg"}], "sample")
