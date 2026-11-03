# Annotation, driver separation and targeted training

The immediate goal is a small, correctly labeled development pilot, followed by
training on suitable examples. The existing State Farm test and video demo are
complete; neither establishes accurate frontal-camera distraction recognition.
Do not repeat those jobs to obtain different scores. Keep DMD unscored while the
revised models and evaluation protocol are being developed.

## 1. What is ready now

Annotation tool: **VGG VIA 3.0.13**, downloaded locally. VIA is an offline browser
application supporting video time segments; no Docker, account or video upload is
needed. [Official download/documentation](https://www.robots.ox.ac.uk/~vgg/software/via/),
[official VIA3 description](https://github.com/ox-vgg/via/blob/master/via-3.x.y/README.md).

The prepared kit is `data/annotations/starter/`:

| Item | Purpose |
| --- | --- |
| `START_HERE.html` | Links to the local annotator and contact sheets |
| `via_video_annotator.html` | The standalone annotation application |
| `via_driver_events_template.json` | Preset event labels, with no invented annotations |
| `identity_review.csv` | 126 canonical video filenames/hashes with identity fields blank |
| `contact_sheets/review_001.jpg` through `review_021.jpg` | Three thumbnails per clip for manual identity review |
| `exports/` | Save your working annotation projects here |
| `label_definitions.json` | Definitions for the preset labels |

Original videos remain in `data/nitymed/source/`. The older
`data/nitymed/annotation_template.csv` is preserved; use the new kit for this
workflow rather than maintaining two competing sets of annotations. The local
tool and thumbnails are ignored by Git; only setup code, schema and guidance are
committed. No identities, intervals or train/validation assignments were invented.

## 2. Review which clips contain the same driver — manual

1. Open `START_HERE.html` in Chrome and open the contact sheets.
2. Open `identity_review.csv` in a spreadsheet editor. Preserve the filenames,
   SHA-256 values and other existing columns; save back as UTF-8 CSV.
3. Review the three thumbnails for each file. When needed, play the original clip
   to check the person more carefully. Provider identity metadata is preferable
   where available. Do not group people using clothing alone.
4. Assign a consistent internal ID, such as `P001`, to one confirmed participant.
   Every recording of that same person gets that ID. Real names are unnecessary.
   These are human-reviewed group identifiers, not guesses based on filenames.
5. Fill `identity_evidence`, `identity_status` (`confirmed` or `uncertain`) and
   `reviewer`. Example evidence: “Same participant confirmed by reviewing these
   recordings and their three sampled views.” Leave uncertain identities unresolved.
6. Fill `session_id` only if the recording session is actually known; otherwise
   leave it blank. Do not invent one session per file.
7. For the pilot, confirm the three selected participants first. Review the
   remaining identity rows before claiming participant-independent splits across
   the collection; unresolved rows stay excluded. You do not need to annotate
   all 126 videos' events before the pilot.

The same person may have different clothes, glasses, expressions or lighting.
Conversely, similarly dressed people are not necessarily the same participant.
Contact sheets help review; they do not automatically establish identity.
Unresolved or potentially duplicated identities must be excluded or conservatively
kept in the same leakage group pending review, not split as distinct drivers.

The failed-demo clip, `P1042748_720.mp4`, is marked `recorded_demo`. Keep that
participant in the development/training group once identified, including their
other clips. It must not become an unseen internal test participant. Exposure is
also recorded in `data/demo/development_usage.json`.

The CSV durations are nominal frame-count/FPS estimates for navigation, not
ground-truth event timing. Mark intervals on the actual video timeline in VIA;
the importer must check container timestamps/durations before frame extraction.

## 3. Start with three pilot clips — manual

Choose the failed-demo clip and two clips with **different confirmed drivers**.
Aim to include a clearly visible yawn and reviewed non-yawning behavior. If those
examples are not present in the first choices, select appropriate development
clips after identity review. Do not select clips using predictions from the model.
This tiny pilot verifies annotation quality and export compatibility; it is not
enough to claim 90% performance or useful unseen-driver accuracy.

For each clip, annotate yawning first. Add distraction labels only when the
activity and necessary hand/object/passenger context can actually be established.
Eye-state annotation can follow later; the existing RT-BENE eye model already
provides a baseline. Do not spend hours marking every blink before the pilot is
checked.

## 4. Open and configure VIA — manual

1. Open `data/annotations/starter/via_video_annotator.html` in Chrome. The local
   copy has already been downloaded; it does not need a server.
2. Hover over toolbar icons to see their names. Click the folder icon labeled
   **Open a VIA Project** and select `via_driver_events_template.json`.
3. Click the plus-circle icon labeled **Add Audio or Video File in Local Computer**.
   Select your first pilot MP4 from `data/nitymed/source/`. You can add the other
   two pilot clips to the same project.
4. Confirm that the video plays and the temporal label attribute is `event_label`.
   Use the preset options rather than creating spelling variants.
5. Work from the original, unannotated videos. Do not annotate the model's
   overlaid preview; its predictions can bias the review.

If reopening a saved local project leaves a video unavailable, use the file's
**File Source** selector to reselect that original MP4. Browser local-file handles
are not permanently embedded in project JSON. Do not recreate the project or
discard existing annotations to reconnect media.

## 5. Mark time intervals — manual

VIA's [video annotator help](https://www.robots.ox.ac.uk/~vgg/software/via/demo/via_video_annotator.html)
documents the timeline and keyboard controls. In the downloaded version:

1. Press **Space** to play/pause. Watch the complete clip once for context.
2. Seek to the start of a clearly observed event and pause.
3. Press **a** to add a temporal segment. Select the segment and set `event_label`.
4. Move/drag its left and right boundaries to the actual event start/end. Zoom
   the timeline to check boundaries; left/right arrows step through frames.
   **Shift+a** updates the nearest edge to the current time, so always verify
   which edge changed rather than assuming it sets the end.
5. Add an evidence/uncertainty note when useful. Use half-open `[start,end)`
   intervals in our eventual imported dataset: the end is the first excluded time.
6. Mark reviewed negative periods explicitly. Unmarked time stays unknown; it
   must never automatically become a negative or safe-driving label.
7. Different tasks may overlap, such as a yawn and closed eyes. Contradictory
   labels for the same task must not overlap. Split uncertain/occluded portions
   out of definite positive/negative segments.

No bounding boxes are required for this first event-annotation pass. We will
derive face/eye crops using the existing landmark pipeline and verify crop
quality separately.

## 6. Use these label rules

| VIA label | What to mark |
| --- | --- |
| `yawn_yes` | A visible yawn: clear yawning mouth/jaw onset through return toward baseline; distinguish it from speech/laughter |
| `yawn_no` | A watched interval with no yawn; include reviewed normal mouth motion and distinguishable speech/laughter |
| `eyes_open` / `eyes_closed` | Visible eye state; a blink can be closed, but that does not establish sleep |
| `distraction_none_visible` | Enough visible context to rule out the four supported target activities; not proof of attention, gaze or global safety |
| `phone_use` | Visible phone-use evidence |
| `drinking` | Visible drinking action |
| `other_object` | A supported radio/reaching/grooming/object activity; specify which one in notes |
| `passenger_interaction` | Evidence of passenger-directed interaction; talking, mouth opening or a yawn alone is insufficient |
| `*_uncertain` | Task-specific ambiguity; exclude from supervised training/scoring |
| `*_not_visible` | Required evidence is obscured/out of view; exclude for that task |

Do not label `distraction_none_visible` solely because no passenger is visible.
If hands/objects needed to rule out phone or object use are hidden, use uncertainty
for that task. Record reliable supplied context, such as a confirmed staged action,
in notes. The historical State Farm name “Safe driving” only describes this
five-class mapping; it must not become a claim of road attention or no drowsiness.

Do not turn an entire “yawning” website category into positive frames. Do not
label every mouth opening as a yawn, every downward look as a nod, or every
forward-facing head as gaze-on-road. Do not diagnose microsleep from eye closure.
Nodding/gaze require an agreed protocol and suitable supervision before adding
those labels. Keep uncertain cases rather than forcing a class.

## 7. Save and submit the pilot — manual

1. Use VIA's **Save current VIA Project** floppy-disk button. Browser Ctrl+S saves
   a webpage and is not the annotation-project save operation.
2. Save/move the downloaded JSON to:
   `data/annotations/starter/exports/pilot_v01.json`.
3. Save your edited `identity_review.csv` in the starter folder.
4. Optionally export a CSV through **Import or Export Annotations** for human
   inspection, but retain the full project JSON as the primary annotation source.
5. Tell the coding agent: “Pilot annotations are ready.” Provide unresolved
   identity questions and any missing label options.

Stop after these three clips until the pilot passes inspection. We will check
that actual local filenames survive export, label options decode correctly, time
intervals fit the video, positives/negatives are explicit, identities are supported
and overlap rules are consistent. Do not invent intervals to fill a template.

## 8. Fill the distraction-data gap before fine-tuning

NITYMED may be useful for yawning and reviewed non-passenger examples. It does
not establish that we have frontal-view positives for all five distraction
classes. More copies of the same clips or more epochs do not fill missing classes.

After the pilot, we will inventory counts by participant, class, domain and
reviewed duration. Genuine passenger/phone/object/drinking examples need to appear
in the intended camera view, including validation participants. If they are absent,
choose one of these concrete ways to obtain them before claiming a five-class fix:

- Obtain an appropriately licensed frontal-cabin action dataset with usable
  participant identities and matching actions; we must verify access/labels first.
- Make permitted staged development recordings in a **parked vehicle**, with
  multiple participants and the intended camera view. Record known normal,
  phone-use, drinking, defined object activity and genuine passenger-interaction
  examples. Have an actual passenger for that condition and record the action
  instructions. Keep repeated takes under the same driver ID. These are staged
  examples, not evidence of real-road performance.

Do not collect every frontal clip as a negative and every State Farm clip as a
positive for one class; the model could learn dataset/camera identity instead of
the activity. Insufficient class/view coverage is a real blocker, not a reason to
fabricate labels or train on DMD. No additional dataset is required to start the
three-clip pilot or the yawning data review.

## 9. Freeze driver-wise splits — coding agent, after identity review

The agent will generate a saved participant manifest before model comparisons:

1. Group all clips of each confirmed participant together. Resolve exact and
   near duplicates across recordings; keep connected uncertain identities together
   or out of independent evaluation until resolved.
2. Force the previously debugged demo participant into training/development.
3. Assign the other participants once using a documented fixed seed, targeting
   approximately 70% training / 15% validation / 15% internal test by driver.
   Preserve original State Farm and RT-BENE split memberships.
   If reliable evidence links a person across datasets, preserve that identity
   grouping too; do not introduce a known held-out driver into new training data.
4. Require usable class support and multiple independent validation/test drivers.
   Aim for at least three drivers in each evaluation group if available; this is a
   practical starting target, not a guarantee or universal sufficiency rule.
5. Save assignments and hashes; all extraction, crops and augmentations inherit
   the parent participant's split. Never randomly split neighboring frames.

The three-clip pilot only exercises the workflow. If the reviewed collection has
too few distinct people/classes, we must expand data or limit the claim; a one-driver
validation result is not a convincing generalization result. Do not reshuffle
groups repeatedly to obtain better scores. DMD remains separate external test data.

## 10. Convert reviewed annotations to training data — coding agent

This adapter is the next implementation after pilot exports are checked; it is
**not already provided by the existing State Farm training notebook**.

We will validate hashes, actual media timing, names, identities, intervals, same-task
overlaps and label vocabulary before extracting samples. For frame classifiers,
start with sparse sampling around 1–2 FPS plus reviewed boundary examples, cap
repeated frames per event/driver and balance actual class/driver diversity. Keep
the original full-rate timelines for later event evaluation; sparse frames alone
cannot measure brief blinks or temporal alert behavior.

Mapping will preserve separate task supervision:

| Reviewed annotation | Training target |
| --- | --- |
| Supported distraction interval | Existing project IDs 0, 1, 2, 3, 5 |
| `yawn_yes` / `yawn_no` | New binary yawning target: yawning / not-yawning |
| `eyes_open` / `eyes_closed` | Existing binary eye-state target |
| Unreviewed, uncertain, not visible | Missing label for the affected task, masked out |

“Not yawning” does not mean globally alert. Do not rename the eye classifier's
closed-eye output to yawning. Preserve simultaneous labels rather than forcing
every frame into one whole-driver state. A future shared model must mask missing
labels per head rather than treating them as negatives.

## 11. Run the two targeted training stages — agent prepares, user runs Kaggle

### A. Adapt distraction to the intended camera

Start from the saved ResNet-18 checkpoint and fine-tune on the authorized State
Farm training images plus the reviewed frontal training group. Keep the five-class
head and its mapping. Mix source/target examples and driver/class contributions
so a long clip does not dominate. Use restrained, label-preserving augmentation.

A starting recipe to register before the new experiment is a smaller backbone
learning rate (for example 0.00001), head rate 0.0001, at most ten epochs and
early stopping after three unimproved validation checks. These are proposed
starting settings, not results or an already implemented training job. Compare
against the frozen baseline on the same validation groups. Define the selection
criterion beforehand, such as equally weighted source-domain and frontal-domain
validation macro-F1, and report per-class recall/precision in each domain.
Do not select on the known State Farm test or DMD results.

### B. Train a yawning classifier

Build a separate two-class head, initially using an ImageNet-pretrained lightweight
backbone such as MobileNetV3-Small and reviewed face/lower-face crops. Include
positive yawns and explicit non-yawn speech/laughter/normal examples where the
annotation can distinguish them. Mask occluded/uncertain crops. Sample by driver
and event instead of treating neighboring frames as independent evidence.

Train only on training participants; choose epochs/recipe using the fixed yawning
validation participants and report macro-F1, yawn precision/recall and confusion.
An open mouth is not automatically a yawn. The existing eye model remains a
separate baseline; it does not already provide this behavior.

For each stage the agent will prepare a readable local notebook, a verified data
bundle and the code version. The user will upload/update the private Kaggle inputs,
connect the current GPU Jupyter runtime, run inspection with training disabled,
then enable **one** full training job. The remote runtime cannot read the laptop's
E: drive. After completion, save Ctrl+S and recover/verify the exact checkpoint
locally before ending the session. Do not reuse the old notebook with new labels
until its adapter, task mapping and inference loading have been implemented.

## 12. Temporal alerts, proposed model and final testing — agent + reviewed data

1. Integrate the verified yawning head alongside distraction/eye state. Evaluate
   frame outputs on source validation before adding temporal persistence.
2. On fully reviewed validation videos, choose confidence/persistence settings
   and a fixed event-matching protocol. Measure false alerts against actual reviewed
   video duration and time-to-alert against annotated event onset; missing labels
   are not confirmed negatives. Current 0.7/1-second demo settings are unvalidated.
3. The proposal's shared-backbone/multiple-head model is still an additional
   implementation/training comparison. Train it with missing-label masking and
   the same driver groups, then compare against the separate-model/rule baselines.
   The existing two-model demo does not already satisfy that proposed architecture.
4. Freeze the selected checkpoints, preprocessing, mappings and event settings.
   Evaluate the new internal test groups and aligned DMD external data once under
   that protocol. Do not use final-test scores to select more recipes.
5. Report source-versus-target performance, class errors, driver differences and
   CPU latency. Our previous CPU preview was slower than 25 FPS; measure again for
   the actual final pipeline instead of assuming real-time performance.
6. Prepare a held-out sample for **you to share with the professor**, with accurate
   labels and a reproducible inference command. The professor does not need to
   supply the test data. Record unsupported nodding/gaze or other incomplete goals
   honestly if adequate data/protocols cannot be obtained.

## Fastest critical path and what to do now

| Order | You | Coding agent | Gate |
| --- | --- | --- | --- |
| Now | Review identities, annotate 3 pilot clips, save JSON | Starter kit ready | Export/labels checked before scaling |
| Next | Resolve ambiguity and annotate additional selected drivers | Import validation, class/domain inventory, fixed splits | Suitable positive/negative coverage |
| Then | Provide missing frontal action examples if needed | Prepare distraction/yawn bundles and notebooks | Independent validation support |
| Training | Run the supplied Kaggle jobs and save outputs | Verify checkpoints and validation reports | No test-based selection |
| Final | Review agreed samples and share held-out sample with professor | Video/event integration, shared-model comparison, DMD alignment/test/report | Frozen protocol and honest scope |

Start with the pilot. Full annotation time depends on video complexity and identity
uncertainty; the approximately 98 minutes of source video will take longer than
98 minutes to label and review. Adding many frames cannot replace diverse drivers
or missing classes, and neither this plan nor more training guarantees 90%.

## Setup verification

The starter was generated for all 126 canonical local MP4s, creating 21 contact
sheets with three samples per video and blank identity columns. The VIA application
copy matches the downloaded Oxford artifact byte-for-byte. Three focused tests
pass for empty/unlabeled templates, preservation and source immutability; Ruff
passes. No event annotation, identity assignment, split or training job has been
created by this setup. A new checkout can regenerate the kit with
`python -m scripts.prepare_annotation_starter` after downloading the official VIA
HTML to `.cache/annotation_tools/`; the command refuses existing annotation work.
