# Driver Distraction and Drowsiness Monitoring

AIT DLCV project. Phase 0 scaffold only: no trained model or inference yet.
The supplied proposal defines [project scope](docs/SCOPE.md): a lightweight shared
CNN with six distraction classes and four drowsiness states, temporal driver alerts,
and an event log for fleet review. Evaluation focuses on unseen drivers, glasses,
low light, and a held-out second dataset. The eight-week plan locks scope in week 2
and delivers a demo, generalization report, and reusable code/splits/label map.
Read [PLAN.md](PLAN.md) before Phase 1. Public data is car-cab footage;
**no trucking validation** is claimed. Eye crops cannot validate full-driver states.

## Windows setup
Python 3.11 required. From this folder:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Python 'C:\path\to\Python311\python.exe'
.\.venv\Scripts\python.exe scripts/check.py
.\.venv\Scripts\python.exe -m ruff check .
```
Setup installs local Phase 0 dependencies and records versions in
`docs/environment-phase0.txt`. ML extras are deferred to Phase 1 for CPU/GPU wheel
selection and MediaPipe API verification. No global installation required.
GNU Make is optional: `make setup BASE_PYTHON=python`, `make test`, `make lint`.
For Linux use `make test PYTHON=.venv/bin/python` after creating a Python 3.11 venv
and installing `pip install -e '.[dev]'` within it.
`make train ARGS=--fast`, `make eval`, `make demo` explicitly fail as placeholders.
The empty-suite wrapper accepts no collection only while no test files exist.
Docker currently runs that setup check; the demo image comes in Phase 5.

## Privacy and reproducibility
Keep datasets in ignored `data/`, outputs in `results/`, secrets in environment
variables. Never commit face imagery or credentials. Fixed seeds, saved subject
splits, YAML configs and logged package versions are required. Dataset access has
not been verified; see [dataset register](docs/DATASETS.md).
