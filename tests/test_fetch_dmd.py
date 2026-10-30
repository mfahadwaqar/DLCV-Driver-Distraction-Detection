"""Secret-safe DMD intake regression checks, using tiny synthetic archives only."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from src.data import fetch_dmd as dmd

NAME = "dmd-dataset-drowsiness-gA-5.tar.gz"


def link(name):
    return f"https://datasets.vicomtech.org/{dmd.ARCHIVES[name][0]}/{name}?token=PRIVATE"


def secret():
    return "\n".join(link(name) for name in dmd.ARCHIVES if name not in dmd.KNOWN_SHA256)


def archive_bytes(member="dmd/gA/5/s5/annotation.json"):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        info = tarfile.TarInfo(member)
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))
    return raw.getvalue()


class Opener:
    def __init__(self, payload, length=None):
        self.payload = payload
        self.length = str(len(payload)) if length is None else str(length)

    def open(self, request, timeout):
        response = io.BytesIO(self.payload)
        response.headers = {"Content-Length": self.length}
        return response


def test_exact_allowlist_and_required_subset():
    assert len(dmd.parse_urls(secret())) == 3
    assert len(dmd.parse_urls(secret() + "\n" + link(next(iter(dmd.KNOWN_SHA256))))) == 4
    for bad in (
        secret().replace("https://", "http://", 1),
        secret().replace("datasets.vicomtech.org", "datasets.vicomtech.org.evil", 1),
        secret().replace("/di21-", "/../di21-", 1),
        secret().replace("?token=PRIVATE", "#PRIVATE", 1),
        secret() + "\n" + link(NAME),
        link(NAME),
    ):
        with pytest.raises(dmd.IntakeError) as error:
            dmd.parse_urls(bad)
        assert "PRIVATE" not in str(error.value)


def test_streamed_atomic_download_and_secret_free_inventory(tmp_path):
    payload = archive_bytes()
    record = dmd.download_archive(NAME, link(NAME), tmp_path, Opener(payload))
    assert (tmp_path / NAME).read_bytes() == payload
    assert not (tmp_path / (NAME + ".part")).exists()
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["gzip_crc_and_tar"] == "passed"
    assert record["file_types"] == {".json": 1}
    assert "PRIVATE" not in json.dumps(record)


@pytest.mark.parametrize("kind", ["truncated", "bad_crc", "unsafe_path", "html", "wrong_sha"])
def test_bad_download_never_published(tmp_path, monkeypatch, kind):
    payload = archive_bytes()
    length = len(payload)
    if kind == "truncated":
        payload = payload[:-5]
    elif kind == "bad_crc":
        payload = payload[:-8] + b"\x00" * 8
    elif kind == "unsafe_path":
        payload = archive_bytes("dmd/gA/5/../../escape.json")
        length = len(payload)
    elif kind == "html":
        payload = b"<html>access expired</html>"
        length = len(payload)
    else:
        monkeypatch.setitem(dmd.KNOWN_SHA256, NAME, "0" * 64)
    with pytest.raises(dmd.IntakeError):
        dmd.download_archive(NAME, link(NAME), tmp_path, Opener(payload, length))
    assert list(tmp_path.iterdir()) == []


def test_transport_error_does_not_echo_url(tmp_path):
    class BrokenOpener:
        def open(self, request, timeout):
            raise OSError(request.full_url)

    with pytest.raises(dmd.IntakeError) as error:
        dmd.download_archive(NAME, link(NAME), tmp_path, BrokenOpener())
    assert "PRIVATE" not in str(error.value)
    assert error.value.__suppress_context__


def test_redirect_denied_without_echoing_location():
    with pytest.raises(dmd.IntakeError) as error:
        dmd.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil/?PRIVATE")
    assert "PRIVATE" not in str(error.value)


def test_existing_archive_revalidated_without_network(tmp_path):
    (tmp_path / NAME).write_bytes(archive_bytes())
    assert dmd.download_archive(NAME, link(NAME), tmp_path)["file_members"] == 1


def test_local_execution_blocked(monkeypatch):
    real_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: False if str(p) == "/kaggle"
                        else real_is_dir(p))
    with pytest.raises(dmd.IntakeError, match="local large downloads are disabled"):
        dmd.run()


def test_notebook_contains_current_source_and_no_outputs():
    notebook = json.loads(Path("notebooks/kaggle_dmd_test_data.ipynb").read_text())
    source = Path(dmd.__file__).read_text()
    assert notebook["cells"][1]["source"] == source.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_crc_fixture_is_corrupted():
    payload = archive_bytes()
    with pytest.raises(gzip.BadGzipFile):
        gzip.decompress(payload[:-8] + b"\x00" * 8)
