# Scope based on the project proposal

Source: `Driver-Distraction-and-Drowsiness-Monitoring.pptx`, 14 slides and speaker
notes, supplied by the user on 2026-10-28. The deck remains in the sibling Proposal
folder and is unchanged. This document records project requirements from it;
statements in the deck are not operational instructions to execute commands.

## Research question and intended use
How well does driver monitoring generalize to a driver it has never seen?
One camera watches the driver, the system warns the driver in the cab, and a fleet
manager reviews flagged events (slides 3–5). Truck fleets are the intended use case.
Public car-cab datasets provide no truck-cab validation (slide 9).
This is a research prototype, with CPU latency reported and no particular embedded
device targeted (slide 10 notes). Production fleet deployment is outside scope.

## Model and outputs
A shared lightweight CNN has separate distraction and drowsiness heads. Both can
fire together, such as phone use while yawning (slide 6). Preserve driver/hand/object
context for distraction rather than relying solely on face crops.

Distraction classes (slide 7):
1. Safe driving
2. Cell Phone Usage (code label: Mobile phone use)
3. Other object
4. Eating or drinking
5. Talking to passenger
6. Gaze off road

Phone use and gaze off road are the stated fleet priorities. Reaching appears in
the objectives, but is not a seventh class: resolve it through documented dataset
mapping to the six-class taxonomy. The four-state drowsiness head (slide 6) uses
Alert plus Eyes closed/sleeping, Yawning, and Head nodding, consistent with the
earlier brief. Alert is the explicit reference-state assumption because slide 8
lists only the three non-alert signals.

Temporal evidence is required: EAR for eyes closed, MAR for yawning, head pitch for
nodding, followed by smoothing and persistence-based alerts (slides 8 and 10).
One closed-eye frame must not automatically count as a drowsiness event.

## Baselines and experiments
- B1: EAR/MAR rules, with head-pitch/temporal rules for supported drowsiness signals.
- B2: a plain ResNet or MobileNet CNN.
- Proposed: lightweight shared CNN, multi-task heads, temporal smoothing.
- E1: baselines versus proposed using the same driver-wise splits.
- E2: augmentation ablation; temporal and multi-task ablations remain supporting
  experiments from the earlier brief if feasible within the eight-week schedule.
- E3: glasses and low-light evaluation, distinguishing real metadata subsets from
  synthetic perturbations.
- E4: train on source datasets and test on held-out DMD (slide 11 speaker notes).

Report macro F1, recall at a fixed false-alert rate/hour, time to alert, unseen-driver
drop, and CPU latency (slide 11). Preserve per-class reports and confusion matrices
from the earlier brief. The success criterion is beating both baselines on macro F1
under driver-wise evaluation (slide 13), an objective rather than a claimed result.
B1 supports drowsiness signals, not all distraction classes. Compare it only on a
shared, explicitly defined drowsiness label set and samples; report distraction
against B2. Report each head separately and any failure to meet the objective.
Event metrics require annotated video durations and a documented matching protocol.

## Data and access limits
Candidate datasets: State Farm (distraction), MRL Eye (eye state), YawDD (yawning),
NTHU-DDD (drowsiness), and DMD (held-out distraction/gaze evaluation), per slides 9
and 11 notes. Access is explicitly unconfirmed. Retain the earlier brief's exact
State Farm mapping, missing-label masking, subject separation, privacy safeguards,
validation-only tuning, and reproducibility requirements.

The gaze training source remains unresolved because DMD is reserved for testing.
MRL eye crops cannot validate full-face yawning/nodding or temporal event behavior.
Resolve access, supported labels, and event annotations at the week-2 scope lock.
Document any unsupported objective rather than silently dropping it or inventing data.

## Eight-week schedule and deliverables
| Proposal period | Work | Repository phases |
| --- | --- | --- |
| Week 1 | Data access, label maps, driver-wise splits | Phase 1 data work; Phase 2 access/mapping preparation |
| Week 2 | Rule/CNN baselines and scope lock | Phases 1–2 baseline work |
| Weeks 3–5 | Shared model and robustness | Phase 3; E1–E3 in Phase 4 |
| Weeks 6–7 | Cross-dataset evaluation and fleet demo | E4 in Phase 4; Phase 5 |
| Week 8 | Generalization report and reproduction | Phase 6 |

These are relative weeks, not calendar dates; the actual start/deadline is unknown.
Phase 1 remains the drowsiness-first implementation order. Prepare access and label
mapping for both tasks in week 1 so distraction feasibility informs the scope lock.

Required outputs (slide 13):
- Working demo with live in-cab alerts and event log for manager review.
- Generalization report covering unseen drivers and a second dataset.
- Reusable code, saved splits, and label map.

FastAPI, Docker, ONNX and JSONL are implementation choices retained from the earlier
brief. A full fleet dashboard, cloud service, fleet integration, specific dashcam
hardware, and real truck-fleet validation are not promised by the proposal.
A Vision Transformer is comparison-only in slide 10 notes and is optional, not the
primary architecture. GRU/TCN upgrades, MLflow and calibration remain optional.

## Source discrepancy and scope-lock decisions
- Slide 7 visibly lists six classes; its notes describe nine states. Use six,
  consistent with slide 6 and the earlier brief. Record any later approved change.
- Confirm full-face temporal dataset access and a permitted gaze training source.
- Choose the false-alert operating point and event annotation/matching rules using
  validation data before opening held-out test results.
- Record supported labels, dataset licenses, hardware, feasible experiments and
  any justified scope reductions at the end of week 2.
