"""User-run NITYMED 720p batch downloader with a dedicated authorized browser session.

Install: .venv/Scripts/python.exe -m pip install playwright
         .venv/Scripts/python.exe -m playwright install chromium
Discover: .venv/Scripts/python.exe scripts/download_nitymed.py
Download: .venv/Scripts/python.exe scripts/download_nitymed.py --download
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from contextlib import contextmanager
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path

PAGE = "https://datasets.esdalab.ece.uop.gr/download-files/"
HOST = "datasets.esdalab.ece.uop.gr"
ROOT = Path(__file__).resolve().parents[1]


class ListingParser(HTMLParser):
    """Read anchor text and its explicitly labeled tab, without guessing endpoint IDs."""

    def __init__(self):
        super().__init__()
        self.nodes = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag, "attrs": dict(attrs), "text": "",
                "parents": list(self.stack)}
        self.nodes.append(node)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        for node in self.stack:
            node["text"] += data

    def anchors(self):
        identifiers = {n["attrs"]["id"]: n for n in self.nodes if "id" in n["attrs"]}
        anchors = []
        for node in self.nodes:
            if node["tag"] != "a" or "href" not in node["attrs"]:
                continue
            panel = next((p for p in reversed(node["parents"])
                          if p["attrs"].get("role") == "tabpanel"), None)
            title = (identifiers.get(panel["attrs"].get("aria-labelledby"))
                     if panel else None)
            anchors.append({"text": node["text"], "href": node["attrs"]["href"],
                            "resolution": title["text"].strip() if title else "",
                            "panel_id": panel["attrs"].get("id", "") if panel else ""})
        return anchors


def public_anchors():
    opener = urllib.request.build_opener(DatasetRedirects())
    with opener.open(PAGE, timeout=90) as response:
        checked_url(response.url)
        if response.status != 200:
            raise ValueError("Public listing did not return HTTP 200.")
        body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError("Unexpectedly large public listing.")
        parser = ListingParser()
        parser.feed(body.decode("utf-8"))
    return parser.anchors()


@contextmanager
def browser_session(direct):
    if direct:
        yield None
        return
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError("Install Playwright or use --direct for the public listing.") from error
    with sync_playwright() as playwright:
        yield playwright.chromium.launch_persistent_context(
            str(ROOT / ".cache/nitymed-browser"), headless=False
        )


def checked_url(url):
    parts = urllib.parse.urlsplit(url)
    if (
        parts.scheme != "https" or parts.hostname != HOST or parts.port not in (None, 443)
        or parts.username or parts.password
    ):
        raise ValueError("A download points outside the approved HTTPS dataset host.")
    return url


def select_links(anchors):
    """Use explicit 720p anchor labels, including hidden tab panels; never guess URLs."""
    links, seen = [], set()
    for anchor in anchors:
        label = " ".join(anchor.get("text", "").split())
        resolution = anchor.get("resolution", "")
        if not re.search(r"(?:HDTV[ _-]*720|HD[ _-]*720|720p)(?!\d)",
                         resolution or label, re.IGNORECASE):
            continue
        if not re.search(r"(?<![A-Za-z])(?:MALE|FEMALE)(?![A-Za-z])", label, re.IGNORECASE):
            continue
        url = checked_url(urllib.parse.urljoin(PAGE, anchor.get("href", "")))
        parts = urllib.parse.urlsplit(url)
        if parts.path.rstrip("/") == "/download-files" and not parts.query:
            continue
        if url not in seen:
            links.append({"label": label, "url": url, "resolution": resolution,
                          "panel_id": anchor.get("panel_id", "")})
            seen.add(url)
    return links


def labeled_group(label):
    """Report only explicitly named categories; link counts do not prove file coverage."""
    sex = re.search(r"(?<![A-Za-z])(female|male)(?![A-Za-z])", label, re.I)
    action = re.search(r"(?<![A-Za-z])(microsleep|yawning)(?![A-Za-z])", label, re.I)
    return (sex[1].lower(), action[1].lower()) if sex and action else None


class DatasetRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        checked_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def file_kind(prefix):
    if prefix[4:8] == b"ftyp":
        return ".mp4"
    if prefix.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return ".zip"
    raise ValueError("Server returned neither MP4 nor ZIP; possible access/error page.")


def safe_filename(headers, response_url, label, extension):
    message = Message()
    message["Content-Disposition"] = headers.get("Content-Disposition", "")
    name = message.get_filename() or urllib.parse.unquote(
        urllib.parse.urlsplit(response_url).path.rsplit("/", 1)[-1]
    )
    if (
        not name or Path(name).suffix.lower() != extension
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith((".", " "))
        or name.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10)),
        }
    ):
        raise ValueError("No safe original server filename; inspect the response manually.")
    return name


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verified_previous(link, results, output):
    for result in reversed(results):
        if result.get("url") != link["url"] or result.get("status") not in {
            "downloaded", "duplicate-content"
        }:
            continue
        name = result.get("file", "")
        if not name or Path(name).name != name or "/" in name or "\\" in name:
            continue
        local = output / name
        if local.is_file() and sha256(local) == result.get("sha256"):
            return result
    return None


def download(link, context, output, index):
    url = checked_url(link["url"])
    cookies = context.cookies([url]) if context is not None else []
    headers = {
        "User-Agent": "NITYMED-academic-download/1.0",
        "Referer": PAGE,
        "Accept-Encoding": "identity",
        "Cookie": "; ".join(f"{cookie['name']}={cookie['value']}" for cookie in cookies),
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=".nitymed-", suffix=".part", dir=output)
    os.close(descriptor)
    partial = Path(temporary_name)
    try:
        request = urllib.request.Request(url, headers=headers)
        opener = urllib.request.build_opener(DatasetRedirects())
        with opener.open(request, timeout=90) as response:
            checked_url(response.url)
            if response.status != 200:
                raise ValueError("Expected a complete HTTP 200 response.")
            prefix = response.read(64)
            extension = file_kind(prefix)
            name = safe_filename(response.headers, response.url, link["label"], extension)
            length = response.headers.get("Content-Length")
            expected = int(length) if length else None
            if expected is not None and (expected < 0 or expected > 2_000_000_000):
                raise ValueError("Invalid or excessive Content-Length; inspect it manually.")
            size = len(prefix)
            with partial.open("wb") as stream:
                stream.write(prefix)
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2_000_000_000:
                        raise ValueError("File exceeds the 2 GB limit; inspect it manually.")
                    stream.write(chunk)
        if expected is not None and size != expected:
            raise ValueError("Incomplete response: downloaded size differs from Content-Length.")
        digest = sha256(partial)
        target = output / name
        if target.exists():
            if sha256(target) == digest:
                return {"label": link["label"], "url": url, "file": name, "bytes": size,
                        "sha256": digest, "status": "duplicate-content"}
            raise ValueError("Original filename conflicts with existing content; inspect manually.")
        # Exclusive creation preserves existing files even if another run publishes first.
        created = False
        try:
            with target.open("xb") as destination, partial.open("rb") as source:
                created = True
                shutil.copyfileobj(source, destination)
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
        return {"label": link["label"], "url": url, "file": target.name, "bytes": size,
                "sha256": digest, "status": "downloaded"}
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download after link discovery")
    parser.add_argument("--direct", action="store_true",
                        help="Use public HTML and downloads without a browser or cookies")
    parser.add_argument("--output", type=Path, default=ROOT / "data/nitymed/source")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with browser_session(args.direct) as context:
        try:
            if args.direct:
                anchors = public_anchors()
            else:
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(PAGE, wait_until="domcontentloaded", timeout=60000)
                input("Complete authorized access in the browser, then press Enter here. ")
                checked_url(page.url)
                listing = ListingParser()
                listing.feed(page.content())
                anchors = listing.anchors()
            links = select_links(anchors)
            if not links:
                raise ValueError("No explicitly labeled 720p links found. No downloads started.")
            print(f"Discovered {len(links)} distinct 720p links.")
            groups = {group for link in links if (group := labeled_group(link["label"]))}
            required = {(sex, action) for sex in ("male", "female")
                        for action in ("yawning", "microsleep")}
            if required - groups:
                missing = ", ".join(f"{sex} {action}" for sex, action in sorted(required - groups))
                print(f"Unconfirmed categories in anchor labels: {missing}.")
            print("Category labels and link counts do not establish full dataset coverage.")
            print("Names such as ALL/FULL are treated as ordinary links, not assumed archives.")
            results_path = args.output / "download_results.json"
            results = (json.loads(results_path.read_text(encoding="utf-8"))
                       if results_path.exists() else [])
            if not isinstance(results, list) or any(not isinstance(r, dict) for r in results):
                raise ValueError("Invalid download manifest; inspect it before continuing.")
            failures = 0
            for index, link in enumerate(links, 1):
                print(f"[{index}/{len(links)}] {link['label']}", flush=True)
                if not args.download:
                    continue
                if verified_previous(link, results, args.output):
                    print("  Already downloaded and verified.", flush=True)
                    continue
                try:
                    result = download(link, context, args.output, index)
                    result.update({key: link[key] for key in ("resolution", "panel_id")})
                    results.append(result)
                    print(f"  {result['status']}: {result['file']}", flush=True)
                except Exception as error:
                    failures += 1
                    # Avoid printing URLs, cookies or third-party response bodies.
                    print(f"  Failed ({type(error).__name__}); continuing to the next link.",
                          flush=True)
                    results.append({"label": link["label"], "url": link["url"], "status": "failed",
                                    "error_type": type(error).__name__})
                temporary = results_path.with_suffix(".json.part")
                temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
                temporary.replace(results_path)
            print("Finished. Inspect download_results.json for failures; rerun to retry them."
                  if args.download else "Discovery only. Add --download to fetch these files.")
            if failures:
                parser.exit(1, f"{failures} download(s) failed.\n")
        finally:
            if context is not None:
                context.close()


if __name__ == "__main__":
    main()
