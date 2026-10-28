# Dataset sources and access status

Checked 2026-10-28. No files downloaded, authenticated access tested, or data uploaded.
Training platform: Kaggle Notebooks. Use original subject metadata, not random
pre-split mirrors. Dataset availability does not itself establish permission to
upload it to Kaggle, even privately.

| Dataset | Official source | Use | Verified status / next step |
| --- | --- | --- | --- |
| State Farm | https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data | Distraction; labeled train images only | Official listing found, subject to competition rules. User login/required rule acceptance and authenticated file access remain to be checked. Preserve driver_imgs_list.csv. |
| MRL Eye | https://mrl.cs.vsb.cz/eyedataset | Eye-open/closed baseline, glasses/lighting subsets | Official page timed out. Author paper confirms subject metadata; current download, license and exact filename format remain unverified. |
| YawDD | https://doi.org/10.21227/e1qm-hb90 | Full-face yawning footage | Official IEEE DataPort search record says free IEEE login required, membership unnecessary. Direct DOI fetch blocked by crawler rules. Terms, size, identities and event annotations still need inspection after login. |
| NTHU-DDD | https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ | Candidate full-face temporal drowsiness data | Official endpoint timed out; index returned gateway error. Current agreement/form, licensing and download process could not be verified. |
| Vicomtech DMD | https://dmd.vicomtech.org/ | Held-out cross-dataset evaluation only | Official site/repository readable. Download not tested. Use current availability and license, not old mirrors. |

## Important DMD update
Official repository: https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
Its 2026 revision says only real-car RGB remains available; simulator, IR and depth
were removed. Listed usable subjects: 1, 5, 6, 7, 9, 10, 13, 14, 23, 28, 29, 33,
36, 37. Its license section states CC BY-NC-ND 4.0. Confirm permissions before
transferring source videos or derived frames to Kaggle. Do not rely on a badge alone.
Older official access guidance describes a form, Box account, terms acceptance and
an emailed link; this is historical guidance, not a verified current workflow:
https://www.vicomtech.org/en/news/detail/562_vicomtech-develops-the-driver-monitoring-dataset-dmd-an-opensource-driver-monitoring-system-now-available-for-academic-and-scientific-use

## Kaggle workflow
- Attach official State Farm competition data if account access allows it.
- For external datasets, obtain authorized original downloads and verify permission
  for Kaggle hosting/transfer before attaching private data. Never publish raw faces.
- Keep original subject IDs, archive checksums, source versions and saved split files.
- Configure notebook input paths; save checkpoints/metrics/config/version logs as
  run outputs. Record the actual GPU/runtime, without assuming GPU quotas or versions.
- Ask before downloads over 5 GB. No download/upload has happened in this task.
- If access is pending, build clearly labeled mock loaders.

## Coverage gaps
MRL crops cannot validate yawning, nodding or temporal alerts. Full-face datasets
need usable identities and event annotations. State Farm has no Gaze off road class
under the requested mapping; DMD remains held out, so another permitted gaze-training
source is still needed at scope lock. Do not substitute a mirror that loses subjects.

MRL primary-source paper: https://mrl.cs.vsb.cz/publications/ISVC18_for_reviews.pdf
