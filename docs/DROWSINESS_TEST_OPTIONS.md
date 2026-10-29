# External drowsiness test data: access and remaining manual work

Checked 2026-10-29 against official sources. No large archive was downloaded,
registration submitted, or message sent. These are candidates, not acquired test
sets. Existing local inspection is documented in PHASE1_DATA.md.

## Practical decision

No newly verified source currently satisfies all three requirements without
manual work: independent participants, eye/yawn/nod event labels, and accessible
authorized video. Keep the RT-BENE held-out result as an internal eye-state test;
it is not a cross-dataset or temporal-alert result.

The smallest practical next step is to use the already authorized NITYMED access
to obtain previously unseen participants, resolve their true recording identities,
and manually annotate continuous videos. Existing four clips remain QA/development
material if they informed thresholds, preprocessing, or design. If no identity
mapping is available, report a clip-level demonstration only. Do not invent IDs
from clip numbers or infer cross-dataset identity from faces.

For a ready-made richer label set, retry NTHU's official portal manually. Treat
access, current agreement, archive size, and native annotations as unverified until
the original package is inspected. Do not replace it with a random frame mirror.

## Candidate comparison and exact manual actions

| Source | What is verified | Manual action and suitability |
|---|---|---|
| [NITYMED official site](https://datasets.esdalab.ece.uop.gr/) | Official description: 21 drivers, yawning and microsleep videos, 25 fps; about 700 MB at 720p or 1.6 GB at full HD; Creative Commons Attribution and citation requested. Public page does not provide a usable participant-to-recording map or interval labels. | Reuse the access already obtained with AIT details; do not register twice. Download selected unseen participants' full clips from the authorized page after identity verification. If access has expired, the public Access Dataset form asks name, affiliation, academic/business email. Annotate eye closure, yawns, nods actually present, and normal/talking/look-away negatives. Full face/video supports temporal testing; eye crops alone support eye state only. |
| [NTHU-DDD official portal](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/) | The official endpoint still returns 502 to this research client. Current download instructions, license and package size are not verified. | Open manually on the user's network. Inspect any current academic request/agreement and original subject/annotation files before deciding. Do not promise a particular form or email process. Full video is the candidate for eyes/yawn/nod timing; official test labels may have separate availability constraints. |
| [UTA-RLDD official page](https://sites.google.com/view/utarldd/home) and its [official Drive folder](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl) | 60 participants, 180 approximately 10-minute RGB videos, 111.3 GB, five participant folds. Labels 0/5/10 describe self-reported predominant alert/low-vigilance/drowsy state per video. Only 36 participants permit publishing faces; citation required. No broad redistribution license established here. | Open Drive manually; inspect small fold/identity metadata and access before selective download. Preserve pseudonymous participant IDs across all three clips. Not recommended as the minimal next download: coarse labels cannot establish eye/yawn/nod onset or time to alert; independent event annotation is needed. Videos were recorded at home/university, not driving. |
| [DROZY official repository](https://orbi.uliege.be/handle/2268/191620), [format documentation](https://www.drozy.ulg.ac.be/), [license](https://orbi.uliege.be/bitstream/2268/191620/5/DROZY-LicenseAgreement-20180718.pdf) | ORBi lists DROZY.zip at 2.46 GB. Source documentation describes SUBJECT-TEST files, NIR videos, timestamps and KSS scores for 14 participants; landmark annotations are not eye/yawn/nod ground truth. Some sessions are missing and some recordings have 15 fps; do not assume constant 30 fps. | Read/accept the source agreement before obtaining the archive. It restricts use to research/evaluation and prohibits redistribution and modification; only designated images are publishable. Obtain clarification before derived-image redistribution/cloud transfer. Suitable for NIR domain evaluation with added independent event annotation, rather than a ready-made RGB event benchmark. |
| [YawDD official DOI](https://doi.org/10.21227/e1qm-hb90) and [original author paper](https://www.site.uottawa.ca/~shervin/pubs/CogniVue-Dataset-ACM-MMSys2014.pdf) | The paper describes mirror/dashboard recordings with normal, talking/singing and yawning behavior. Local participant PDF was already inspected; female/male IDs are separate namespaces and publication permission varies by recording. Existing repo access notes describe 4.94 GB and an IEEE login; current crawler access is blocked. | Use the existing signed-in IEEE account, inspect current terms and participant metadata, and selectively retrieve original videos if permitted. The previously inspected AWS route needs separate account/subscription access; do not assume free page login gives AWS access. Video action names are not frame-level yawn labels. Keep this dataset out of test if it is used for yawning training. |
| [YawDD+ author release](https://opensource.silicon-austria.com/mujtabaa/yawdd), linked by [the authors' 2026 paper](https://arxiv.org/html/2512.11446v3) | New candidate: authors describe human-verified frame-level yawning/non-yawning labels. Official GitLab project exists, but this client sees only its loading shell; archive contents, size, participant IDs, frame-to-video mapping and dataset license remain unverified. | Open the author repository manually and inspect README/license plus annotation filenames before downloading. Confirm mappings back to original participants and continuous video timestamps. Paper's CC BY license is not proof of dataset redistribution permission. Useful only for yawning; it does not supply eye closure or head-nod truth. |
| [MRL Eye official page](https://mrl.cs.vsb.cz/eyedataset/) | Still times out. Existing loader filename expectations are synthetic-fixture tested, not verified on an original archive. | If the page works manually, obtain original archive with subject filenames and current terms. This would be a small independent eye-state test, but cannot test full-face/temporal behavior. No download-ready fallback is claimed. |

## Test protocol before opening results

1. Freeze preprocessing, models, EAR/MAR/pitch thresholds, persistence, cooldown,
   false-alert operating point and event matching using development/validation data.
2. Record official source/version, hashes, participant mapping and the reserved
   participant list. Keep all resolutions and clips of one participant together.
   Keep test data out of training, augmentation selection and threshold tuning.
3. Annotate independent ground truth with source video ID, pseudonymous true
   participant ID, start/end timestamps, signal, visibility/uncertainty and reviewer.
   Do not use the tested EAR/MAR/pitch rule as its own ground truth. Label overlap
   explicitly; closed eyes and yawns can coexist, so one exclusive class per frame
   may lose truth.
4. Preserve continuous negative footage and original timestamps to report false
   alerts/hour and time to alert. Distinguish ordinary blinks from sustained closure
   and distinguish intentional gaze/head movement from nodding. Exclude uncertain
   intervals by a frozen rule, not according to model errors.
5. Report only supported signals. A yawning-only test cannot validate nodding,
   and KSS or microsleep clip names cannot provide event onset. Report uncertainty
   across participants, not frames treated as independent people.

These are research videos; none establishes generalization to truck cabs or
operational fleet safety.
