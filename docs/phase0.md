# Phase 0 verification — 2026-10-28

Created Git repository, Python 3.11.9 `.venv`, requested directories, YAML config,
privacy ignore rules, setup/check commands, Makefile, and Docker setup scaffold.
Installed local development dependencies; versions in environment-phase0.txt.
No dataset downloaded and no ML training performed.

Measured checks:
- `python scripts/check.py`: pytest 8.4.2 collected 0 tests in 0.03s. Empty-suite
  wrapper returned 0. This is setup verification, not passing functional tests.
- `python -m ruff check .`: all checks passed.
- `python -m pip check`: no broken requirements found.
- Python compilation of src/ and scripts/: succeeded.
- YAML parsed; split proportions sum to 1.
- Git ignore checks covered data/example.csv, .env, data/face.jpg and checkpoints/model.pt.

Not verified: dataset access/licenses, ML dependencies/API compatibility, model
behavior, GPU, Docker build, or Make execution (Make unavailable on PATH).
Train/eval/demo are explicit failing placeholders. No performance results exist.
The initial restricted-network install failed; the permitted network retry succeeded.

Next: user review of PLAN.md before Phase 1. Need dataset access/locations,
training hardware, grading rubric/deadline, and appropriate full-face temporal data.
