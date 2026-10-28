# Driver Distraction and Drowsiness Monitoring

See [Kaggle workflow](docs/KAGGLE.md) for committed-code submission, run status,
and downloading saved experiment outputs to this local repository.

AIT DLCV project. A first real-data RT-BENE **eye-state** baseline is complete;
the full distraction and drowsiness system is still in progress. See the
[RT-BENE result](docs/RT_BENE_RESULTS.md), [Phase 1 status and commands](docs/phase1.md), and
[data handoff](docs/PHASE1_DATA.md). Unit-test fixtures are synthetic and do not
establish detection accuracy.
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
Setup installs local setup dependencies and records versions in
`docs/environment-phase0.txt`. For Phase 1 tests install CPU wheels locally, then
the extra dependencies:
```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,kaggle,phase1]"
```
Local versions are recorded in docs/environment-phase1-local.txt. Training runs
on Kaggle; local optimizer steps are synthetic software tests only.
GNU Make is optional: `make setup BASE_PYTHON=python`, `make test`, `make lint`.
For Linux use `make test PYTHON=.venv/bin/python` after creating a Python 3.11 venv
and installing `pip install -e '.[dev]'` within it.
`make train ARGS="--data-root /kaggle/input/ACTUAL-SOURCE/mrlEyes_2018_01 --device cuda --fast"`
submits the first small B2 run once a permitted source is attached. `make eval`
runs B1 feature evaluation (provide --features and --output via ARGS).
`make demo` remains a Phase 5 placeholder. The Docker scaffold checks setup only;
install the Phase 1 extra explicitly before running the expanded test suite.

## Privacy and reproducibility
Keep datasets in ignored `data/`, outputs in `results/`, secrets in environment
variables. Never commit face imagery or credentials. Fixed seeds, saved subject
splits, YAML configs and logged package versions are required. RT-BENE access
has been verified; other dataset access varies. See [dataset register](docs/DATASETS.md).

Training will run on Kaggle Notebooks. Local development and CPU tests/demo stay
in this repository. Record actual notebook runtime/package/GPU versions per run.
See docs/DATASETS.md for official sources and access status.
