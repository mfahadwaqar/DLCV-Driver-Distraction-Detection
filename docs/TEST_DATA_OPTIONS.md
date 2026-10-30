# External testing data and manual access

Verified 2026-10-29 against official sources. No video download, form submission,
email, purchase, training, split change, or model evaluation was performed here.

## Access update: 2026-10-30

The user's AIT inbox contains access emails for all three DMD packages. The
signed drowsiness README opened in Chrome and describes frame intervals for
eye states, blinking and yawning (with/without a hand), with participant/session
structure and OpenLABEL JSON. Activities were acted. This verifies documented
coverage; actual participant archive annotation completeness remains unchecked.
The emailed links specify a 48-hour validity. Signed access URLs are not stored
in this repository. Browser archive-download attempts timed out; no completed
archive download was verified. Local intake folder: `data/dmd/incoming/` (ignored).

## Recommended order

Keep **DMD as the primary external test dataset**, as required by SCOPE.md.
Request its current distraction and gaze packages first; request drowsiness if
full-face temporal evaluation is also planned. **Drive&Act is the strongest
verified accessible video backup**: its original annotation archive actually
contains participant IDs and frame intervals. It is simulator data and cannot
replace DMD without explicitly recording the E4 scope change.

Neither dataset implies 85% accuracy. External testing measures domain shift;
results and supported labels must be reported honestly.

## DMD: current access, not historical Box instructions

The [official repository](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset)
limits the 2026 release to real-car RGB and subjects 1, 5, 6, 7, 9, 10, 13, 14,
23, 28, 29, 33, 36, 37. It describes OpenLABEL/VCD annotations and CC BY-NC-ND
4.0 licensing. Use current MP4s and annotations, not removed simulator/IR/depth
material. Package-level label coverage still needs inspection after access.

The [official homepage](https://dmd.vicomtech.org/) lists temporal actions,
phone/texting, drinking, passenger interaction, gaze regions, yawning and
microsleep. It requires academic use and age 18+. Its old 25 TB figure is not a
verified size for today's reduced packages. Current download size is unknown.

Exact manual steps, verified from the official public portal client and its
read-only dataset metadata response:

1. Open [Distraction RGB](https://opendatasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/a268f10c),
   [Gaze & Hands](https://opendatasets.vicomtech.org/di21-dmd-dataset-gaze/8433939e),
   or [Drowsiness](https://opendatasets.vicomtech.org/di21-dmd-dataset-drowsiness/99e2fbd7).
2. Enter **Email Address**, **Full Name**, and **Organisation** using your own
   academic details. Read the linked terms/privacy policy before checking consent.
3. Click **Request access to dataset** yourself. The success screen says access
   instructions will arrive at the provided email. No login/account requirement
   was shown in the inspected request flow; eventual delivery/download was not tested.
4. Follow that email; first inspect the README, file inventory, sizes, participant
   IDs and matching annotation files. Avoid downloading everything automatically.

All three public metadata responses were active and described free academic
access. Source for the inspected form implementation:
[official portal client](https://opendatasets.vicomtech.org/static/js/main.50f843a3.js).
No personal details were sent. This supersedes the unverified historical Box flow
in DATASETS.md; it does not prove that access emails arrive or every file works.

## Drive&Act: verified metadata, video not acquired

The [official page](https://driveandact.com/) permits research use and links public
downloads; its [access update](https://www.driveandact.com/post/access-restrictions/)
says login requirements were removed. The [authors' paper](https://driveandact.com/publication/2019_iccv_drive_and_act/2019_iccv_drive_and_act.pdf)
describes 15 participants, static simulator recordings, manual/automated modes,
frame annotations and subject-separated benchmarks. RGB uses the co-driver
Kinect view; video timestamps/frame indices must be checked against its annotations.

Official HTTP HEAD checks returned 200 on 2026-10-29:

| Asset | Official download | Bytes | Status |
| --- | --- | ---: | --- |
| Activity annotations | [iccv_activities_3s.zip](https://driveandact.com/dataset/iccv_activities_3s.zip) | 8,814,535 | Downloaded for metadata inspection only |
| RGB videos | [kinect_color.zip](https://driveandact.com/dataset/kinect_color.zip) | 5,256,569,061 | Not downloaded; exceeds the project's 5 GB approval threshold |

Local inspection archive: `data/inspection/driveandact/iccv_activities_3s.zip`.
SHA-256: `d200ad6a00f2f374cf4cfe54fcf6565bb17eb25961829ae71eb75d89033a760a`.
Do not add this third-party archive to Git.

The actual `activities_3s/kinect_color/midlevel.chunks_90.csv` has 10,333 rows,
participants 1–15, and fields `participant_id,file_id,annotation_id,frame_start,
frame_end,activity,chunk_id`. Object-level files additionally contain `object`
and `location`. These are chunks, not independent drivers or complete independent
events: reconstruct original annotations using file/annotation IDs and validate
interval boundaries before time-to-alert or false-alert/hour scoring.

Observed labels include `interacting_with_phone`, `talking_on_phone`, `eating`,
`drinking`, object manipulation, display/laptop use, looking behind and
`sitting_still`. Proposed mappings are hypotheses requiring a frozen mapping
review: phone labels to Mobile phone use; eating/drinking to Eating or drinking;
supported object interactions to Other object. `sitting_still` is not automatically
Safe driving, and looking/searching is not an exact calibrated gaze-off-road
label. No passenger-talking, yawning or nodding label was found in that CSV.
Mask unsupported classes rather than fabricate negatives.

Manual route: open the official page, inspect **Annotations → Activities** first,
then select **A-Column co-driver → RGB** only after approving the archive size and
storage location. No account or request form is currently needed for those links.
Research use is explicit; permission to rehost raw/derived material on Kaggle is
not established by the landing page. Use local evaluation unless hosting rights
are resolved; do not publish face frames or mirrors.

## Evaluation safeguards and remaining gaps

- Seal external test data from training, augmentation selection, threshold tuning,
  checkpoint selection and label-rule experimentation. Tune on source validation
  drivers, freeze configuration, then evaluate once and report failures.
- Reserve all DMD views/clips of each subject together. Do not call random frame
  splits unseen-driver evaluation. Check cross-dataset identity overlap as far as
  metadata permits; disjoint dataset names alone do not prove distinct people.
- Before scoring, inspect the released annotations for usable positives,
  negatives, overlaps, unannotated intervals, durations and action definitions.
  Fix event matching and the false-alert operating point using validation only.
- DMD's ND condition prohibits distributing modified material under the stated
  license; do not infer private Kaggle transfer permission. Record any specific
  permission obtained before transferring videos or derived frames.
- YawDD and NTHU remain candidates already documented in PHASE1_DATA.md, with
  access/terms/event coverage unresolved. RT-BENE is already used for source
  training and cannot serve as the independent external dataset. Existing NITYMED
  samples lack verified subject linkage/event intervals, so they remain inspection
  clips until those gaps are resolved. None establishes truck-cab performance.
