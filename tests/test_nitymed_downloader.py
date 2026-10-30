import io
from types import SimpleNamespace

import pytest

from scripts.download_nitymed import (
    PAGE,
    ListingParser,
    checked_url,
    download,
    file_kind,
    labeled_group,
    safe_filename,
    select_links,
    sha256,
    verified_previous,
)


def test_selects_only_explicit_720_links_and_deduplicates_targets():
    links = select_links([
        {"text": "MALE_HDTV720_1", "href": "/?download=1"},
        {"text": "MALE_HDTV720_ALL", "href": "/?download=1"},
        {"text": "FEMALE_HDTV720_FULL", "href": "/?download=2"},
        {"text": "MALE_FULL_1", "href": "/?download=3"},
    ])
    assert len(links) == 2
    assert [link["label"] for link in links] == ["MALE_HDTV720_1", "FEMALE_HDTV720_FULL"]


def test_rejects_foreign_hosts_and_error_pages():
    with pytest.raises(ValueError):
        checked_url("https://example.com/file.mp4")
    with pytest.raises(ValueError):
        file_kind(b"<html>Access required</html>")
    assert file_kind(b"\x00\x00\x00\x20ftypisom") == ".mp4"
    assert file_kind(b"PK\x03\x04sample") == ".zip"


def test_filename_does_not_invent_a_video_name_from_label():
    with pytest.raises(ValueError):
        safe_filename({}, PAGE + "?download=1", "MALE_HDTV720_1", ".mp4")


@pytest.mark.parametrize("name", ["../a.mp4", "a\\b.mp4", "CON.mp4", "bad?.mp4"])
def test_rejects_unsafe_server_filename(name):
    with pytest.raises(ValueError):
        safe_filename({"Content-Disposition": f'attachment; filename="{name}"'},
                      PAGE, "unused", ".mp4")


def test_preserves_original_filename():
    assert safe_filename({"Content-Disposition": 'attachment; filename="Real name_09.mp4"'},
                         PAGE, "ALL", ".mp4") == "Real name_09.mp4"


def test_coverage_does_not_default_unknown_actions_to_yawning():
    assert labeled_group("FEMALE_HDTV720_ALL") is None
    assert labeled_group("MALE_HDTV720_MICROSLEEP_1") == ("male", "microsleep")
    assert labeled_group("FEMALE_YAWNING_HD720_1") == ("female", "yawning")
    assert not select_links([{"text": "FEMALE_HD7200_1", "href": "/?download=1"}])


def test_tab_resolution_selects_microsleep_links_without_720_in_anchor():
    parser = ListingParser()
    parser.feed('''<div id="title-a" role="tab">HDTV720</div>
        <div id="panel-a" role="tabpanel" aria-labelledby="title-a" hidden>
        <a href="/download/867/">MALE_MICRO_HDTV1 (312 downloads)</a></div>
        <div id="title-b" role="tab">FULL HD</div>
        <div role="tabpanel" aria-labelledby="title-b">
        <a href="/download/999/">FEMALE_HDTV720_1</a></div>''')
    links = select_links(parser.anchors())
    assert len(links) == 1
    assert links[0]["panel_id"] == "panel-a"
    assert links[0]["resolution"] == "HDTV720"
    assert links[0]["url"].endswith("/download/867/")


def test_resume_requires_same_url_and_verified_content(tmp_path):
    path = tmp_path / "real.mp4"
    path.write_bytes(b"original")
    record = {"label": "same", "url": PAGE + "?id=1", "file": path.name,
              "sha256": sha256(path), "status": "downloaded"}
    assert verified_previous({"url": record["url"]}, [record], tmp_path) == record
    assert verified_previous({"url": PAGE + "?id=2"}, [record], tmp_path) is None
    path.write_bytes(b"changed")
    assert verified_previous({"url": record["url"]}, [record], tmp_path) is None


def mock_response(monkeypatch, payload, declared_length=None):
    response = io.BytesIO(payload)
    response.url = PAGE + "?id=1"
    response.status = 200
    response.headers = {"Content-Disposition": 'attachment; filename="real.mp4"',
                        "Content-Length": str(len(payload) if declared_length is None
                                              else declared_length)}
    monkeypatch.setattr("scripts.download_nitymed.urllib.request.build_opener",
                        lambda *args: SimpleNamespace(open=lambda *args, **kwargs: response))
    return SimpleNamespace(cookies=lambda urls: [])


def test_download_preserves_real_name_and_existing_files(monkeypatch, tmp_path):
    payload = b"\x00\x00\x00\x20ftypisom" + b"fixture" * 20
    context = mock_response(monkeypatch, payload)
    sentinel = tmp_path / ".nitymed-1.part"
    sentinel.write_bytes(b"existing partial")
    link = {"label": "MALE_HD720_ALL", "url": PAGE + "?id=1"}
    result = download(link, context, tmp_path, 1)
    assert result["file"] == "real.mp4"
    assert result["url"] == link["url"]
    assert (tmp_path / "real.mp4").read_bytes() == payload
    assert sentinel.read_bytes() == b"existing partial"
    context = mock_response(monkeypatch, payload + b"different")
    with pytest.raises(ValueError, match="conflicts"):
        download(link, context, tmp_path, 1)
    assert (tmp_path / "real.mp4").read_bytes() == payload
    assert sorted(p.name for p in tmp_path.iterdir()) == [".nitymed-1.part", "real.mp4"]


def test_truncated_response_is_not_published(monkeypatch, tmp_path):
    context = mock_response(monkeypatch, b"\x00\x00\x00\x20ftypisom", declared_length=100)
    with pytest.raises(ValueError, match="Incomplete"):
        download({"label": "ALL", "url": PAGE + "?id=1"}, context, tmp_path, 1)
    assert list(tmp_path.iterdir()) == []
