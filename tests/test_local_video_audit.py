"""Tiny synthetic archives exercise safe metadata intake and full gzip verification."""

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "local_video_audit", Path(__file__).parents[1] / "scripts/audit_local_video_data.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def make_archive(path, members):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, payload in members:
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    path.write_bytes(gzip.compress(raw.getvalue()))


@pytest.mark.parametrize(
    "name", ["../escape.json", "/escape.json", "C:/escape.json", "a\\..\\escape.json"]
)
def test_reject_escape(tmp_path, name):
    with pytest.raises(ValueError):
        audit.safe_target(tmp_path, name)


def test_only_metadata_extracted_and_intervals_counted(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    payload = {
        "openlabel": {
            "actions": {
                "0": {
                    "type": "eyes_state/close",
                    "frame_intervals": [{"frame_start": 2, "frame_end": 4}],
                }
            }
        }
    }
    make_archive(
        archive,
        [
            ("dmd/gA/1/s5/ann.json", json.dumps(payload).encode()),
            ("dmd/gA/1/s5/video.mp4", b"not-decoded"),
        ],
    )
    output = tmp_path / "out"
    result = audit.audit_archive(archive, output)
    assert result["annotations"][0]["actions"][0]["labeled_frames_inclusive"] == 3
    assert len(result["members"]) == 2
    assert len(list(output.rglob("*.json"))) == 1
    assert not list(output.rglob("*.mp4"))
    assert result["integrity"].endswith("passed")


def test_checks_gzip_crc_after_tar_end_marker(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    make_archive(archive, [("README.txt", b"metadata")])
    content = bytearray(archive.read_bytes())
    content[-8] ^= 1
    archive.write_bytes(content)
    with pytest.raises(gzip.BadGzipFile):
        audit.audit_archive(archive, tmp_path / "out")


def test_nitymed_manifest_mismatch_and_blank_identity(tmp_path):
    source = tmp_path / "data/nitymed/source"
    source.mkdir(parents=True)
    cache = tmp_path / ".cache"
    cache.mkdir()
    output = tmp_path / "inspection"
    output.mkdir()
    video = source / "clip.mp4"
    video.write_bytes(b"synthetic-file-hashed-only")
    record = {
        "file": video.name,
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "bytes": video.stat().st_size,
        "metadata": {"format": {"duration": "1.0"}},
        "decode_returncode": 0,
    }
    (cache / "nitymed-verification.json").write_text(json.dumps({"files": [record]}))
    (source / "download_results.json").write_text(
        json.dumps([{**record, "status": "downloaded", "sha256": "mismatch"}])
    )
    result = audit.verify_nitymed(tmp_path, output)
    assert result["files"][0]["matches_previous_verification"]
    assert not result["all_hashes_match"]
    import csv

    with (tmp_path / "data/nitymed/annotation_template.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["participant_id"] == row["session_id"] == row["event_label"] == ""
    assert row["annotation_status"] == "unreviewed"
