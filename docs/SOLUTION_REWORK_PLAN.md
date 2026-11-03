# Rework: distinguish observations from unsupported action claims

Prepared 2026-11-04 in response to the mirror-check/passenger confusion and
annotation-tool difficulties. This is a proposed design for review before
changing the model taxonomy or runtime alert behavior. Existing checkpoints,
test reports, original annotations and completed recordings are preserved.

## Evidence and limits

- State Farm c9 maps to project class 5, passenger interaction. No mirror-check
  subtype or head-direction supervision enters the current classifier.
- The failed NITYMED development preview was labeled passenger in all 250 frames,
  including forward-facing frames, with approximately 0.967–0.9996 confidence.
  The saved checkpoint reproduces the output; the display mapping is correct.
- A mirror check is a plausible interpretation of some side glances, as the user
  points out. It has not been independently verified as frame-level ground truth.
  This is not only a single head-turn mistake: the constant label also indicates
  a broader camera/domain generalization problem.
- A sideward look alone does not establish passenger interaction. Conversely,
  a passenger can be outside the camera view, so “no detected passenger” must not
  automatically mean mirror check or safe driving.

## Proposed output design

| Layer | What it may report | What it must not infer without support |
| --- | --- | --- |
| Observable cues | Head orientation, visible eye state, visible phone/drink/object evidence | Head orientation is not a gaze target or conversational intent |
| Action hypotheses | Existing action-class scores, with camera/context limitations | A high softmax score is not a verified activity |
| Operational alert | A supported action with source-validation temporal criteria | Every side glance is distraction or passenger interaction |

For a new/unvalidated camera view, retain raw model outputs in diagnostics and
mark unsupported action claims as uncertain. Do not relabel passenger logits as
“mirror check,” and do not force uncertain cases into Safe driving. Suppressing
an unsupported alert is a presentation/decision change, not an improvement in
classifier accuracy; raw errors remain in evaluation reports.

Head-orientation observations would need calibration for the actual camera/view.
They should remain geometric observations, not claimed eye-gaze or mirror-intent
ground truth. A separate person detector alone would not solve intent.

## Data needed for the actual learned fix

Create reviewed development examples in the intended camera view containing:

1. Confirmed normal mirror checks, including cases resembling passenger-directed
   head turns, with source/script evidence where intent cannot be observed.
2. Genuine passenger interaction from comparable views and different drivers.
3. Forward-facing normal behavior, yawns, speech and other negative/confounding
   examples; sufficient hand/object context for distraction labels.
4. Supported phone/drinking/object actions so adaptation does not teach “all
   frontal views are safe.”

Use mirror check as an annotation/context subtype for development, not a silent
seventh proposal class. Map it to the existing no-target-action class only when
other target activities can also be ruled out. Otherwise use missing/uncertain
distraction supervision. Final taxonomy changes require an explicit scope decision.

Keep every participant and their clips in one group. The debugged NITYMED
participant stays in training/development. Preserve source splits and DMD's
external-test role. Do not tune on known State Farm test scores or DMD predictions.

## Staged implementation after output definitions are agreed

1. Update the demo decision/display contract to distinguish raw suggestions,
   supported observations and unknown actions; preserve diagnostic predictions.
   Do not claim mirror detection from the existing five-class checkpoint.
2. Review a small mirror-versus-passenger development pilot with reliable labels
   and positive examples in both training and validation camera distributions.
   NITYMED yawning footage alone does not supply this coverage.
3. Fine-tune with reviewed data plus State Farm training data. Predefine the
   validation criterion and compare against the frozen baseline on identical
   driver groups, including class precision/recall and mirror false positives.
4. Train the yawning task independently using explicit yawn/non-yawn supervision;
   eye closure and yawning remain separate signals. Combine temporally only after
   source-validation event labels exist.
5. Evaluate the frozen revision on independent internal/DMD test data under a
   fixed protocol. Report any remaining uncertainty, errors and CPU limitations.

Acceptance must include both mirror checks not spuriously becoming passenger
alerts and genuine passenger interactions still being recognized. A UI-only
suppression of every passenger output does not meet learned-model acceptance.
The original 76.94% State Farm test result remains historical evidence, not a new
score for this proposed revision.

## Annotation interface repair already prepared

The missing yawn row is consistent with a default/unconfigured VIA project;
there is no current live browser observation proving the exact UI state. The
previous instructions assumed a preset loaded successfully and should not have
kept asking the user to find a row they could not see.

A separate `data/annotations/starter/yawn_labeler.html` now initializes its own
three-option yawning preset on startup: `yawn_yes`, `yawn_no`, `yawn_uncertain`.
It removes the project-import step and leaves the original VIA app, JSON template,
identity CSV and any existing exports unchanged. No identities or intervals were
created. Code/schema tests and the actual VIA data-loader API accept the preset;
this is not a claim that the live Chrome UI was visually verified.

Browser/computer-use tooling currently fails to initialize with “failed to write
kernel assets … path specified.” No browser security setting was changed and no
access restriction was bypassed. The proposed model rework does not depend on the
user continuing with the old annotation screen.
