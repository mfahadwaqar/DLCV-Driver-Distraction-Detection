# DMD local intake

Inspected 2026-10-30. DMD remains held out for the final external test. This was
archive and annotation inspection only: no training, inference, evaluation,
threshold selection, video extraction, download, or duplicate deletion.

## Local archives and integrity

Both local files exist in `data/`:

- `dmd-dataset-drowsiness-gA-1.tar.gz`
- `dmd-dataset-drowsiness-gA-1 (1).tar.gz`

Each is **828,325,787 bytes** (about 790 MiB). Both have SHA-256
`a5b1d475452373e22fd0a99174fdec4ab509d694de6316c2f69b2b28908092b4`.
They are byte-identical copies, so the duplicate adds no coverage. Both were
read completely through gzip without CRC, truncation, or trailer errors;
each expands to 830,791,680 bytes of tar data. A streaming tar walk completed.
This verifies local archive integrity, not completeness against a provider
manifest (none was supplied) or successful video decoding.

Only the 921,495-byte JSON annotation was safely copied from the archive under
ignored `data/dmd/inspection/dmd/gA/1/s5/`. No README exists in this archive.
Local generated inventory, annotation summary, and video header summary are in
`data/dmd/inspection/`; `/data/` is already ignored by Git. Archive paths were
resolved under that destination before metadata writes; no archive links or
video members were extracted.

## Actual scope and inventory

One participant, **subject 1**, group `gA`, session `s5`, recording
`2019-03-14T14;26;17+01;00`. The annotation context says **Car Stopped**, **Rainy**.
This is one recording of one driver, not a multi-driver external benchmark.

All files below share prefix `dmd/gA/1/s5/gA_1_s5_2019-03-14T14;26;17+01;00_`.
The archive also has three directory entries and no other file members.

| Suffix | Bytes | Observed metadata |
| --- | ---: | --- |
| `rgb_mosaic.avi` | 119,412,596 | AVI header: 5,481 frames, 1,280 x 720, rate 744/25 = 29.76 fps |
| `rgb_face.mp4` | 358,613,848 | MP4: 5,480 video samples, 184.139785 s, 29.76 fps |
| `rgb_body.mp4` | 351,833,347 | MP4: 5,427 video samples, 182.358871 s, 29.76 fps |
| `rgb_ann_drowsiness.json` | 921,495 | OpenLABEL schema 1.0.0, seven action types |

MP4 timing and counts came from `moov`/`mdhd`/`stts`/`stsz` container headers
read directly through the archive stream; encoded video was not saved or decoded.
Both MP4s have timescale 11,904 and sample duration 400 ticks. The AVI header
was read in memory. These checks do not establish visual camera synchronization.

## Observed labels and intervals

Counts below use inclusive endpoints (`end - start + 1`). Intervals and per-frame
action references agree exactly; each action's annotation pointer intervals also
match its declared intervals. Eye states partition frames **0–5,479** without
overlap or gaps. The extra frame record 5,480 has no action labels.

| Exact action type | Intervals | Labeled frames | Labeled seconds at 29.76 fps |
| --- | ---: | ---: | ---: |
| `eyes_state/open` | 83 | 3,596 | 120.833 |
| `eyes_state/close` | 28 | 660 | 22.177 |
| `eyes_state/opening` | 92 | 778 | 26.142 |
| `eyes_state/closing` | 92 | 446 | 14.987 |
| `blinks/blinking` | 83 | 1,884 | 63.306 |
| `yawning/Yawning with hand` | 2 | 349 | 11.727 |
| `yawning/Yawning without hand` | 3 | 420 | 14.113 |

Blink intervals overlap eye-state intervals by design; their counts must not be
added as independent frames. Closed-eye intervals last 3–60 frames (about
0.101–2.016 s), while blink intervals last 5–103 frames. An eye closure or blink
is not automatically a microsleep or fatigue ground-truth event. No nodding,
explicit microsleep/fatigue, phone, eating/drinking, passenger interaction,
object interaction, or gaze-region action type occurs in this JSON. Their absence
is **unsupported annotation coverage**, not evidence that every frame is a
negative example for those classes.

All observed yawn intervals are listed below. Time spans are nominal half-open
`[start/fps, (end+1)/fps)` on the face timeline, before any synchronization correction.

| Label | Inclusive frames | Nominal seconds |
| --- | --- | --- |
| Without hand | 1,363–1,560 | 45.800–52.453 |
| With hand | 2,351–2,512 | 78.999–84.442 |
| Without hand | 2,513–2,540 | 84.442–85.383 |
| Without hand | 3,474–3,667 | 116.734–123.253 |
| With hand | 4,408–4,594 | 148.118–154.402 |

The adjacent 2,351–2,512 and 2,513–2,540 segments change hand category with no
gap; five labeled intervals need not mean five independent physiological yawns.
Full eye/blink intervals remain in the original local JSON; the table summarizes
coverage without copying the participant annotations into Git.

## Alignment issues to resolve before scoring

- JSON declares global interval 0–5,480 and contains 5,481 frame records, but the
  face MP4 has 5,480 samples and face stream metadata says `total_frames: 5480`.
  Use valid decoded face indices 0–5,479 after verification; do not blindly index
  the terminal record 5,480. Mosaic header length is 5,481, so it is not a drop-in
  replacement for the face stream.
- Face stream shift is 0. Body metadata declares 5,427 frames and `frame_shift: 54`;
  hands declares 5,407 frames and `frame_shift: 74`. Both count-plus-shift sums
  equal 5,481. Confirm the shift sign/convention with documentation and decoding
  before mapping annotation indices to body/hands frames; equal raw indices are
  not evidence of alignment.
- JSON references `rgb_hands.mp4`, but that member is **absent**. Face and body
  stream URI basenames match present members. Mosaic has no stream definition
  in the JSON. Do not assume the mosaic restores the missing standalone view.
- FPS is 29.76, not 30. Use actual timestamps/rational timing. Decode integrity,
  image visibility, presentation timestamps, start offsets, and camera alignment
  remain unchecked because no video was extracted or decoded.

## Minimal next subset and blockers

Retain this complete subject-1 face recording and its JSON as the smallest
available **held-out drowsiness integration subset**: about 184 s, with eye-open,
closed, transitions, blinks and hand-covered/uncovered yawn annotations. After
freezing the model, preprocessing, class mapping, thresholds and event rules on
source validation only, verify decoding/time alignment and then score supported
labels on the full recording. Preserve contiguous non-yawn and eye-open stretches;
do not cherry-pick only positive clips. Keep all views and sessions of subject 1
together. One driver cannot substantiate broad unseen-driver performance or a
stable false-alerts-per-hour estimate.

The **distraction archive is absent** from the downloaded DMD files, as is a
separate gaze/hands package. E4 distraction coverage remains blocked until the
appropriate archive and matching annotations are available and inspected. A
second copy of the drowsiness archive does not unblock it. A broader final DMD
test also needs additional independently held-out participants; do not use this
sample to select thresholds or label rules and later report it as untouched test
data. No further download or model run was performed during intake.
