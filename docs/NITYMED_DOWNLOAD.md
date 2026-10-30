# NITYMED batch downloader

`scripts/download_nitymed.py` supports the official public listing without browser
dependencies. The user changed the saved site permission to allow browsing and downloads
before live access on 2026-10-31. Public HTML returned HTTP 200; its HDTV720 panels exposed
130 distinct download URLs. The complete batch processed every discovered URL without
failures: 126 downloads and four duplicate-content responses, producing 126 unique
original filenames and 126 unique SHA-256 hashes under `data/nitymed/source`.

Offline integrity verification completed for every unique file:

- Total size: **697,049,508 bytes** (about 664.76 MiB).
- Combined video duration: **5,861.907769 seconds** (97 minutes 41.9 seconds).
- All 126 MP4s: ffprobe confirmed **1280 × 720**; no probe errors.
- Frame rates: 125 clips at 25 fps and one at 24 fps.
- Individual durations: 14.015979 to 146.602 seconds.
- Every file passed full `ffmpeg -v error -xerror -i FILE -f null -` decoding
  with exit code zero and empty error output.
- SHA-256 and byte counts matched every successful manifest record.

The ignored `.cache/nitymed-verification.json` report records each file's hashes,
probe metadata, and decoding result. `download_results.json` records all 130 URL results.
The observed HDTV720 panels contained 49 male-yawning, 60 female-yawning,
12 male-microsleep, and nine female-microsleep links. Each panel's `ALL` or `FULL`
target duplicated an ordinary MP4 in that panel; none was a bulk archive. These
are the site's displayed categories and link counts, not verified clinical labels.
This verifies all URLs exposed in the current 720p listing, without establishing
official dataset-wide recording/subject completeness.

Discover and download public links without Playwright, browser sessions, or cookies:

```powershell
.venv/Scripts/python.exe scripts/download_nitymed.py --direct
.venv/Scripts/python.exe scripts/download_nitymed.py --direct --download
```

For an optional interactive browser session, install:

```powershell
.venv/Scripts/python.exe -m pip install playwright
.venv/Scripts/python.exe -m playwright install chromium
```

These installations have not been performed as part of this work. Discovery opens
a dedicated persistent Chromium profile under `.cache/nitymed-browser` and waits
for the user to complete any authorized access steps:

```powershell
.venv/Scripts/python.exe scripts/download_nitymed.py
```

Review the discovered link labels. Once satisfied, run:

```powershell
.venv/Scripts/python.exe scripts/download_nitymed.py --download
```

The default output is `data/nitymed/source`; `--output PATH` selects another directory.
Run only one downloader against an output directory at a time. Keep its browser profile
and manifest private because they can contain session information or download URLs.

The downloader selects links mentioning male or female from explicitly labeled 720p tabs,
including hidden tab panels. Tab resolution is read through `aria-labelledby`, so
microsleep anchors lacking `720` are included when their tab explicitly says `HDTV720`.
Without tab metadata, explicit 720p anchor text is required. It accepts HTTPS URLs and redirects only on the official
dataset host. It does not click download controls, infer URLs, or solve access challenges.
Sites whose link labels differ from these offline assumptions may require a parser update.

`ALL` and `FULL` are ordinary labels: neither proves that a target is an archive or
contains a full group. Category reporting uses explicit yawning/microsleep text only;
unknown labels remain unconfirmed. Unique link counts do not prove subject, recording,
or complete dataset coverage. ZIP archives are saved without extraction or member counts.

Downloads preserve the original safe MP4/ZIP filename from the response header or URL.
Missing or unsafe original names require manual inspection; participant IDs are never
invented. A different existing file with the same name causes failure rather than replacement.
Temporary files use unique names; existing partial files are untouched.

`download_results.json` records URLs, original names, panel IDs, resolution labels,
byte counts, hashes, and failures.
Reruns skip a file only when its URL and current SHA-256 match a successful manifest
record. Old records without URLs are not used to skip requests. Interrupted or failed
files restart from the beginning; this is completed-file resume, not HTTP range resume.
Failures continue to later links and produce a nonzero exit code at the end.

The downloader rejects HTML/error signatures, non-200 responses, declared-length
mismatches, files over 2 GB, and data without an MP4/ZIP signature. Signatures and local
hashes do not prove playable video, a valid archive, an upstream reference checksum,
720p dimensions, correct annotation alignment, or completeness. Live verification must
inspect the downloaded content and official annotations before claiming any of these.
Training readiness requires confirmed subject/recording identifiers and matching
annotations; downloading files alone does not establish readiness.

Offline verification:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_nitymed_downloader.py -q
.venv/Scripts/python.exe -m ruff check scripts/download_nitymed.py tests/test_nitymed_downloader.py
```
