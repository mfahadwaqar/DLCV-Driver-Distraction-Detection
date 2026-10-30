"""Download selected held-out DMD archives in Kaggle; never extract or evaluate them."""

import gzip
import hashlib
import json
import shutil
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

INPUT_ROOT = Path("/kaggle/working/dmd/raw")
REPORT_ROOT = Path("/kaggle/working/artifacts")
SECRET_NAME = "DMD_SIGNED_URLS"
ARCHIVES = {
    "dmd-dataset-drowsiness-gA-1.tar.gz": ("di21-dmd-dataset-drowsiness", "gA", "1"),
    "dmd-dataset-drowsiness-gA-5.tar.gz": ("di21-dmd-dataset-drowsiness", "gA", "5"),
    "dmd-dataset-drowsiness-gB-6.tar.gz": ("di21-dmd-dataset-drowsiness", "gB", "6"),
    "dmd-dataset-distraction-gA-1.tar.gz": (
        "di21-dmd-dataset-distraction-rgb-ir", "gA", "1"
    ),
}
KNOWN_SHA256 = {
    "dmd-dataset-drowsiness-gA-1.tar.gz":
        "a5b1d475452373e22fd0a99174fdec4ab509d694de6316c2f69b2b28908092b4"
}
MAX_ARCHIVE_BYTES = 30_000_000_000
CHUNK_BYTES = 1024 * 1024


class IntakeError(RuntimeError):
    """A sanitized error safe to display in a saved notebook."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise IntakeError("Provider redirect rejected; refresh the original access link.")


def parse_urls(secret):
    """Accept newline-separated URLs; permit only the four exact provider paths."""
    selected = {}
    for raw in secret.splitlines():
        url = raw.strip()
        if not url:
            continue
        try:
            parts = urllib.parse.urlsplit(url)
            name = parts.path.rsplit("/", 1)[-1]
            expected = ARCHIVES.get(name)
            valid = (
                expected is not None
                and parts.scheme == "https"
                and parts.netloc == "datasets.vicomtech.org"
                and parts.path == f"/{expected[0]}/{name}"
                and parts.query
                and not parts.fragment
                and not any(char.isspace() for char in url)
            )
        except Exception:
            valid = False
        if not valid:
            raise IntakeError("Secret contains an unapproved DMD host/path or malformed link.")
        if name in selected:
            raise IntakeError("Secret contains duplicate archive names.")
        selected[name] = url
    required = set(ARCHIVES) - {"dmd-dataset-drowsiness-gA-1.tar.gz"}
    if not required.issubset(selected):
        raise IntakeError("Secret must include distraction gA-1 and drowsiness gA-5/gB-6.")
    return selected


def inspect_archive(path, name):
    """Stream gzip/tar metadata through EOF, validating CRC without extracting files."""
    _, group, subject = ARCHIVES[name]
    prefix = PurePosixPath("dmd", group, subject)
    files = 0
    member_bytes = 0
    suffix_counts = {}
    with gzip.open(path, "rb") as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as archive:
            for member in archive:
                member_path = PurePosixPath(member.name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or "\\" in member.name
                    or not member_path.is_relative_to(prefix)
                    or not (member.isfile() or member.isdir())
                ):
                    raise IntakeError(f"Unexpected archive member layout: {name}")
                if member.isfile():
                    files += 1
                    member_bytes += member.size
                    suffix = member_path.suffix.lower()
                    suffix = suffix if suffix in {".mp4", ".avi", ".json"} else "other"
                    suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        while decoded.read(CHUNK_BYTES):
            pass  # Read the gzip trailer too; tar EOF alone does not validate CRC.
    if not files or not suffix_counts.get(".json"):
        raise IntakeError(f"Archive lacks files or JSON annotations: {name}")
    return {
        "group": group, "subject": subject, "file_members": files,
        "member_bytes": member_bytes, "file_types": suffix_counts,
        "gzip_crc_and_tar": "passed", "extracted": False,
    }


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(name, url, output, opener=None):
    """Publish an archive atomically only after size, SHA and gzip/tar validation."""
    target = output / name
    partial = output / (name + ".part")
    try:
        if target.exists():
            digest = hash_file(target)
        else:
            opener = opener or urllib.request.build_opener(NoRedirect())
            request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
            with opener.open(request, timeout=120) as response:
                length = response.headers.get("Content-Length")
                length = int(length) if length is not None else None
                if length is not None and not 0 < length <= MAX_ARCHIVE_BYTES:
                    raise IntakeError(f"Invalid provider content length: {name}")
                if length and shutil.disk_usage(output).free < length + 100_000_000:
                    raise IntakeError(f"Insufficient temporary disk space: {name}")
                count = 0
                checksum = hashlib.sha256()
                with partial.open("wb") as stream:
                    while chunk := response.read(CHUNK_BYTES):
                        count += len(chunk)
                        if count > MAX_ARCHIVE_BYTES or (length and count > length):
                            raise IntakeError(f"Download exceeded allowed size: {name}")
                        stream.write(chunk)
                        checksum.update(chunk)
                if not count or (length is not None and count != length):
                    raise IntakeError(f"Download size mismatch: {name}")
                digest = checksum.hexdigest()
        if name in KNOWN_SHA256 and digest != KNOWN_SHA256[name]:
            raise IntakeError(f"Known local SHA-256 mismatch: {name}")
        candidate = target if target.exists() else partial
        inventory = inspect_archive(candidate, name)
        inventory.update({
            "archive": name, "bytes": candidate.stat().st_size, "sha256": digest,
            "checksum_reference": "previous local copy" if name in KNOWN_SHA256
            else "computed only; no provider checksum supplied",
        })
        if candidate == partial:
            partial.replace(target)
        return inventory
    except IntakeError:
        raise
    except Exception:
        # HTTP errors and tracebacks can include the signed request URL.
        raise IntakeError(
            f"Download/integrity check failed: {name}; refresh link if expired."
        ) from None
    finally:
        partial.unlink(missing_ok=True)


def run(secret_client=None):
    """Kaggle entry point. Secrets never enter environment, command lines or reports."""
    if not Path("/kaggle").is_dir():
        raise IntakeError("Run this intake in Kaggle; local large downloads are disabled.")
    try:
        if secret_client is None:
            from kaggle_secrets import UserSecretsClient

            secret_client = UserSecretsClient()
        secret = secret_client.get_secret(SECRET_NAME)
    except Exception:
        raise IntakeError("Enable the DMD_SIGNED_URLS Kaggle Secret for this notebook.") from None
    urls = parse_urls(secret)
    del secret
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in urls.items():
        print(f"Fetching and checking {name}", flush=True)
        records.append(download_archive(name, url, INPUT_ROOT))
        print(f"Verified {name}", flush=True)
    report = {
        "purpose": "held-out test intake only; no extraction, decoding, evaluation or training",
        "archive_storage": str(INPUT_ROOT), "archives": records,
        "raw_archives_persisted_as_output": True,
    }
    report_path = REPORT_ROOT / "dmd_intake_inventory.json"
    temporary = REPORT_ROOT / "dmd_intake_inventory.json.part"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(f"Sanitized inventory saved: {report_path}", flush=True)
    return report
