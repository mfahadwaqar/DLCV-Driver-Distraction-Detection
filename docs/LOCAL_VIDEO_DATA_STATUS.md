# Local video data preparation status

Audited offline on 2026-10-31 with `scripts/audit_local_video_data.py`.
No download, video extraction, video decoding, training, inference, evaluation,
threshold selection, or original-file deletion/move was performed by this audit.
DMD remains **external held-out data only**. NITYMED is not ready for a
participant-independent split or event-level scoring until annotation is completed.

## Reproduce and local artifacts

```powershell
.venv/Scripts/python.exe scripts/audit_local_video_data.py
.venv/Scripts/python.exe -m pytest tests/test_local_video_audit.py -q
```

The audit reads every compressed byte of the six local archives, hashes compressed
content with SHA-256, traverses their tar entries, and drains gzip through EOF to
check CRC/trailer integrity. Only regular JSON/README/license members of at most
32 MiB are eligible for copying, after path containment checks; links are never
extracted. It does not call `extractall`. Synthetic tests cover POSIX/Windows path
escapes, metadata-only extraction, inclusive frame counts, a corrupt gzip trailer,
and NITYMED manifest mismatch/blank identity fields. Seven tests pass; Ruff passes.

Generated files are under ignored `data/dmd/inspection/`:

- `local_video_audit.json`: archive hashes, every tar member/size, annotation
  identities, full action intervals/counts, stream frame counts/shifts, contexts,
  and the fresh NITYMED hash verification.
- One subdirectory per archive containing the original JSON annotations.

The ignored `data/nitymed/annotation_template.csv` contains 126 unreviewed rows
keyed by actual filename and SHA-256, with identity/session/event/time fields
deliberately blank. Existing templates are preserved on rerun so human annotations
are not overwritten. It sits outside the original-video `source/` directory.

The six input archives total **18,004,142,893 bytes**. The metadata copied from
them totals **23,489,084 bytes**, across 12 JSON files. No README/license member was
present. Approximately 232.8 GB disk space remained after the audit (other active
local preparation may also affect free space). Inputs are in `data/dmd/incoming/`.

## DMD archive integrity and coverage

All six passed gzip CRC/trailer verification and complete streaming tar traversal.
These are local integrity checks; they do not establish provider-manifest
completeness or successful video decoding.

| Archive suffix after `dmd-dataset-` | SHA-256 |
| --- | --- |
| `distraction-gA-1.tar.gz` | `10a9e55c2187a95ef1b1eccb5eab44b3708f88ef598f60e9f31b53ed95266c62` |
| `distraction-gA-5.tar.gz` | `5f00d9c1a076935ffab55c749dd2733300d1b8649dd5bfd71d78b456b2b07af0` |
| `distraction-gB-6.tar.gz` | `82f63a7bdfddef847ac6b7854f64c703404298875bab5c8cbf7485e7a01bfd67` |
| `drowsiness-gA-1.tar.gz` | `a5b1d475452373e22fd0a99174fdec4ab509d694de6316c2f69b2b28908092b4` |
| `drowsiness-gA-5.tar.gz` | `90a8977c98b9c2407cc0738f6c9cb5bac155b1e4a184a61adcc328ea18b41e17` |
| `drowsiness-gB-6.tar.gz` | `6d971dfcebd755fb15453c785a0a4ff6f12d948d7eb220f922d6b96acf2402e8` |

The gA/1 drowsiness hash matches the previously inspected copy documented in
[DMD_INTAKE.md](DMD_INTAKE.md). That document describes the earlier one-archive
snapshot; its statements that distraction/additional participants are absent are
superseded by this six-archive inventory.

Three observed participant keys are **gA/1, gA/5, gB/6**. Keep every session,
view, and package for the same participant together. Each distraction archive
contains s1, s2, s3 (three JSONs and 12 videos: face/body/hands/mosaic per session).
Each drowsiness archive contains s5 (one JSON and three videos: face/body/mosaic).
All three drowsiness JSONs reference a standalone hands MP4 that is absent.

| Participant | Session | JSON frame records | Declared face frames |
| --- | --- | ---: | ---: |
| gA/1 | s1 | 6,777 | 6,777 |
| gA/1 | s2 | 13,641 | 13,640 |
| gA/1 | s3 | 1,845 | 1,844 |
| gA/1 | s5 | 5,481 | 5,480 |
| gA/5 | s1 | 7,009 | 7,009 |
| gA/5 | s2 | 12,416 | 12,416 |
| gA/5 | s3 | 1,836 | 1,836 |
| gA/5 | s5 | 5,287 | 5,287 |
| gB/6 | s1 | 6,748 | 6,748 |
| gB/6 | s2 | 14,882 | 14,882 |
| gB/6 | s3 | 1,948 | 1,948 |
| gB/6 | s5 | 5,410 | 5,410 |

Exact observed distraction action types, pooled across these annotations:

- `driver_actions/`: `safe_drive`, `unclassified`, `phonecall_left`,
  `phonecall_right`, `texting_left`, `texting_right`, `drinking`, `radio`,
  `hair_and_makeup`, `reach_side`, `reach_backseat`, `talking_to_passenger`,
  `change_gear`.
- `gaze_on_road/looking_road`, `gaze_on_road/not_looking_road`.
- `hands_using_wheel/both`, `hands_using_wheel/none`,
  `hands_using_wheel/only_left`, `hands_using_wheel/only_right`.
- `talking/talking`, `hand_on_gear/hand_on_gear`.

Exact observed drowsiness types are `eyes_state/open`, `eyes_state/close`,
`eyes_state/opening`, `eyes_state/closing`, `eyes_state/undefined`,
`blinks/blinking`, `yawning/Yawning with hand`, and
`yawning/Yawning without hand`. Undefined eye state occurs only in gB/6 s5
(two intervals, 10 frames) and must not become a negative/open-eye label.

| Drowsiness participant | Closed-eye intervals / frames | Yawn-with-hand intervals / frames | Yawn-without-hand intervals / frames |
| --- | ---: | ---: | ---: |
| gA/1 | 28 / 660 | 2 / 349 | 3 / 420 |
| gA/5 | 30 / 550 | 2 / 257 | 2 / 247 |
| gB/6 | 13 / 176 | 2 / 305 | 2 / 300 |

Frame interval endpoints are inclusive. Overlapping label families are not
independent frames; adjacent yawn categories need not be separate physiological
events. There is no explicit fatigue/microsleep/nodding action type in these
JSONs. No `eating` action appears. Absence of an action type does not establish
that all unannotated frames are confirmed negatives for it.

## DMD timing and remaining gates

All face streams declare shift 0. Body shifts range from 5 to 54 frames; hands
shifts range from 18 to 74. Full per-recording shifts/counts and ISO recording
times are preserved in the ignored inventory. Some declared totals and JSON
frame counts disagree by one frame. Some count-plus-shift sums also differ from
JSON counts; do not infer synchronization or a shift sign from arithmetic alone.

The JSON stream properties expose no FPS/timescale here. Previous direct
container-header inspection established 29.76 fps for **gA/1 s5 only**, with
5,480 face samples versus 5,481 mosaic frames; see DMD_INTAKE.md. This audit
does not propagate that FPS to other recordings, re-decode video, or verify
per-camera timing. Before any authorized final scoring, verify container and
decoded frame counts, actual rational frame rate/PTS, visibility, terminal-frame
handling, and shift convention for each selected view. Resolve missing hands
views explicitly. Freeze preprocessing, label mapping, thresholds, and event
rules using source validation before accessing DMD for model scoring.

## NITYMED verification and manual annotation needs

Fresh SHA-256 hashes and byte counts for all **126 source MP4s** match both the
previous `.cache/nitymed-verification.json` and all **130 successful manifest
records** in `data/nitymed/source/download_results.json` (126 downloaded plus four
duplicate-content records). No unexpected MP4 was found in the source directory.
Total size is **697,049,508 bytes**, duration **5,861.907769 seconds**. Prior full
ffmpeg decoding passed for all 126; that existing result was reused, without
re-decoding. These checks establish the identity of these local files, not the
identity of the people shown.

Complete `data/nitymed/annotation_template.csv` before splitting or
using it as event ground truth:

1. Establish participant identity across recordings using reliable supplied
   identity information or a documented manual review; assign consistent
   `participant_id` only with evidence in `identity_evidence`. Filenames, site
   categories, and distinct hashes are not participant IDs. Leave unresolved
   identity unknown and keep affected clips out of participant-independent claims.
2. Record actual `session_id` where supported; separate unknown sessions from
   invented per-file IDs. Check repeated people/near-duplicate recording content
   across files, even though exact duplicate hashes were already deduplicated.
3. Review each full recording, mark `visibility`, and annotate supported observed
   events with onset/end seconds using the actual video timeline. Duplicate rows
   for multiple events; preserve filename/hash. Use half-open `[start,end)` times.
   Define the event taxonomy and operational onset/end rules before annotating.
4. Review non-event portions too. Add confirmed reviewed negative intervals only
   after inspection; an empty event field remains unknown/unreviewed. Record
   uncertainty/occlusion in notes, annotator, and annotation status, and adjudicate
   ambiguous events. Website yawning/microsleep categories are clip-level context,
   not verified event boundaries or clinical diagnoses.
5. After identity review, create participant-disjoint train/validation/test groups
   for the authorized NITYMED role. All clips/events of one participant remain in
   one group. The template contains no synthetic IDs or inferred event labels.
