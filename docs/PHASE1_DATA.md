# Phase 1 data handoff

Large archives do not have to pass through this local machine just because training
runs on Kaggle. Prefer an authorized direct download inside Kaggle once the access
mechanism is verified. Do not expose account cookies, signed download URLs or tokens
in a public notebook. If direct access is unavailable, agree on a permitted transfer
route first. Source terms still apply to private Kaggle hosting.

Local small-file locations:
- data/yawdd/Participants Information.zip (received and inspected)
- data/yawdd/Participants Information.pdf (extracted locally; kept out of Git)
- data/mrl/ for an original archive or sample filenames, if locally available
- data/models/face_landmarker.task for the official MediaPipe asset (downloaded)

## What was actually inspected
The YawDD PDF has nine pages of recording-level participant metadata. Its public-
picture-sharing flag varies, including between actions for the same participant.
The participant numbers in the female and male tables are separate namespaces.
Do not treat a participant number alone as a globally unique identity. Do not
publish unrestricted collections of example faces. This PDF does not establish
permission to mirror all videos on Kaggle; source use/hosting terms remain to be
checked before transfer. Frame/event labels and cross-recording identities must
be verified against the actual video archive.

MRL's official page and archive endpoint timed out in Chrome on 2026-10-28. No images or
original archive were acquired; exact filename-schema verification remains pending.
The strict parser encodes the expected eight-field format and is tested only on
synthetic fixtures until the original data can be inspected. NTHU's official page
returned `ERR_CONNECTION_REFUSED` in Chrome. Do not present either dataset as
acquired or ready for training. A third-party MRL mirror was not accessed: browser
review blocked it under the user's official-source restriction.

The signed-in IEEE YawDD page is reachable. Its official AWS access dialog lists
`s3://ieee-dataport/open/31591/YawDD.rar.gz` and individual extracted AVI paths,
including `Mirror/Female_mirror/1-FemaleNoGlasses-Yawning.avi`. This confirms that
selective retrieval is technically possible without downloading the whole archive
locally. The [IEEE AWS instructions](https://ieee-dataport.org/help/accessing-ieee-dataport-dataset-files-through-amazon-web-services-aws)
say it needs IEEE account AWS keys and an IEEE DataPort subscription. Those
credentials must not be copied into Git, notebook source, or chat. We have not
verified that this account has AWS access, transferred video to Kaggle, or
validated individual-video labels. The official page says research use is allowed
and public screenshots must obey the per-recording permission flag; it does not
establish permission to publicly rehost the videos.

## Frame manifest for requested-access video datasets
Once data/access/annotations are available, prepare a CSV with:
path,subject_id,dataset,label_distraction,label_drowsiness,metadata
Use relative frame paths, true identities linked across clips, -1 for unavailable
labels, and optional JSON metadata. Never assign every frame of a yawning video a
yawning label: talking/normal intervals require frame or interval annotation.
The adapter rejects missing identity and DMD in Phase 1. It is a manifest adapter,
not a claim that YawDD/NTHU native annotation formats have been implemented.

## Remaining user input
If MRL's official site is unavailable to you too, send an existing original archive
location or an accessible source link retaining original subject filenames and
stated usage terms. No 4.94 GB YawDD local download is required at this stage.
