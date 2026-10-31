"""Offline metadata-only intake: verify archives; never extract/decode video or score models."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

BLOCK = 4 * 1024 * 1024
MAX_METADATA = 32 * 1024 * 1024


def safe_target(root: Path, name: str) -> Path:
    """Reject Windows and POSIX escapes, even when running on another platform."""
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
        raise ValueError(f"Unsafe member path: {name!r}")
    target = (root / Path(*path.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Member escapes metadata root: {name!r}")
    return target


def annotation_summary(payload: dict, name: str) -> dict:
    label = payload.get("openlabel", payload.get("vcd", payload))
    actions = []
    for uid, action in label.get("actions", {}).items():
        intervals = action.get("frame_intervals", [])
        actions.append(
            {
                "id": uid,
                "type": action.get("type"),
                "intervals": intervals,
                "labeled_frames_inclusive": sum(
                    i["frame_end"] - i["frame_start"] + 1 for i in intervals
                ),
            }
        )
    streams = {}
    for key, stream in label.get("streams", {}).items():
        properties = stream.get("stream_properties", {})
        streams[key] = {
            "uri": stream.get("uri"),
            "total_frames": properties.get("total_frames"),
            "sync": properties.get("sync"),
            "timing_properties": {
                k: v
                for k, v in properties.items()
                if any(t in k.lower() for t in ("fps", "rate", "time"))
            },
        }
    parts = PurePosixPath(name).parts
    frames = label.get("frames", {})
    indices = [int(f) for f in frames if str(f).isdigit()]
    return {
        "member": name,
        "schema_metadata": label.get("metadata", {}),
        "group": parts[1] if len(parts) > 3 else None,
        "participant": parts[2] if len(parts) > 3 else None,
        "session": parts[3] if len(parts) > 4 else None,
        "driver_ids": [
            obj.get("name")
            for obj in label.get("objects", {}).values()
            if obj.get("type") == "driver"
        ],
        "frame_record_count": len(frames),
        "frame_index_min": min(indices, default=None),
        "frame_index_max": max(indices, default=None),
        "global_frame_intervals": label.get("frame_intervals"),
        "streams": streams,
        "actions": actions,
        "contexts": label.get("contexts", {}),
    }


def audit_archive(archive: Path, destination: Path) -> dict:
    """Stream the entire gzip to EOF (including CRC trailer), hashing compressed bytes."""
    digest = hashlib.sha256()

    class HashReader:
        def __init__(self, raw):
            self.raw = raw

        def read(self, size=-1):
            data = self.raw.read(size)
            digest.update(data)
            return data

    result = {
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "members": [],
        "annotations": [],
        "metadata_extracted": [],
        "video_extracted": False,
        "integrity": "pending",
    }
    # tarfile stops at end-of-archive; explicitly drain gzip afterward to check its trailer.
    with archive.open("rb") as raw, gzip.GzipFile(fileobj=HashReader(raw)) as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as tar:
            for member in tar:
                target = safe_target(
                    destination / archive.name.removesuffix(".tar.gz"), member.name
                )
                result["members"].append(
                    {
                        "name": member.name,
                        "bytes": member.size,
                        "kind": "file"
                        if member.isfile()
                        else "directory"
                        if member.isdir()
                        else "link_or_other",
                    }
                )
                is_metadata = target.suffix.lower() == ".json" or target.name.lower().startswith(
                    ("readme", "license")
                )
                if not member.isfile() or not is_metadata:
                    continue
                if member.size > MAX_METADATA:
                    raise ValueError(f"Metadata exceeds {MAX_METADATA} bytes: {member.name}")
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable metadata: {member.name}")
                contents = source.read(MAX_METADATA + 1)
                if len(contents) != member.size:
                    raise ValueError(f"Truncated metadata: {member.name}")
                if target.suffix.lower() == ".json":
                    result["annotations"].append(
                        annotation_summary(json.loads(contents), member.name)
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
                result["metadata_extracted"].append(str(target.relative_to(destination)))
        while compressed.read(BLOCK):
            pass
    result.update(sha256=digest.hexdigest(), integrity="gzip CRC/trailer and tar traversal passed")
    return result


def verify_nitymed(root: Path, output: Path) -> dict:
    previous = json.loads((root / ".cache/nitymed-verification.json").read_text(encoding="utf-8"))
    source = root / "data/nitymed/source"
    manifest = json.loads((source / "download_results.json").read_text(encoding="utf-8"))
    successful = [r for r in manifest if r.get("status") in {"downloaded", "duplicate-content"}]
    rows = []
    for record in previous["files"]:
        path = source / record["file"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(BLOCK), b""):
                digest.update(block)
        rows.append(
            {
                "video_file": path.name,
                "sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
                "matches_previous_verification": digest.hexdigest() == record["sha256"]
                and path.stat().st_size == record["bytes"],
                "duration_seconds": record["metadata"]["format"]["duration"],
                "prior_decode_returncode": record.get("decode_returncode"),
            }
        )
    template = root / "data/nitymed/annotation_template.csv"
    template.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a person's edits if rerun; hashes and filenames do not imply identity or events.
    if not template.exists():
        with template.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "video_file",
                    "sha256",
                    "participant_id",
                    "session_id",
                    "identity_evidence",
                    "event_label",
                    "start_seconds",
                    "end_seconds",
                    "visibility",
                    "annotation_status",
                    "annotator",
                    "notes",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "video_file": row["video_file"],
                        "sha256": row["sha256"],
                        "annotation_status": "unreviewed",
                    }
                )
    expected = {r["video_file"] for r in rows}
    observed = {r["video_file"]: r for r in rows}
    manifest_matches = all(
        r.get("file") in observed
        and r.get("sha256") == observed[r["file"]]["sha256"]
        and r.get("bytes") == observed[r["file"]]["bytes"]
        for r in successful
    )
    return {
        "files": rows,
        "unique_files": len(rows),
        "all_hashes_match": all(r["matches_previous_verification"] for r in rows)
        and manifest_matches,
        "successful_manifest_records": len(successful),
        "all_successful_manifest_records_match": manifest_matches,
        "unexpected_source_videos": sorted(
            p.name for p in source.glob("*.mp4") if p.name not in expected
        ),
        "total_duration_seconds": sum(float(r["duration_seconds"]) for r in rows),
        "video_decode_performed_this_run": False,
        "identity_or_events_inferred": False,
        "annotation_template": str(template.relative_to(root)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "data/dmd/inspection"
    output.mkdir(parents=True, exist_ok=True)
    archives = sorted((root / "data/dmd/incoming").glob("*.tar.gz"))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disk_free_bytes_before": shutil.disk_usage(root).free,
        "dmd_role": "external held-out only; no training, tuning, inference or evaluation",
        "archives": [],
        "errors": [],
    }
    for archive in archives:
        print(f"Verifying {archive.name} ({archive.stat().st_size:,} bytes)", flush=True)
        try:
            summary["archives"].append(audit_archive(archive, output))
            print(f"PASS {archive.name}", flush=True)
        except (OSError, EOFError, ValueError, tarfile.TarError) as exc:
            summary["errors"].append({"archive": archive.name, "error": str(exc)})
            print(f"FAIL {archive.name}: {exc}", flush=True)
        (output / "local_video_audit.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    summary["nitymed"] = verify_nitymed(root, output)
    summary["disk_free_bytes_after"] = shutil.disk_usage(root).free
    (output / "local_video_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Audit saved to {output / 'local_video_audit.json'}", flush=True)
    return 1 if summary["errors"] or not summary["nitymed"]["all_hashes_match"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
