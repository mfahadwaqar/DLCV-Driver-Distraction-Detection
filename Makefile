PYTHON ?= .venv/Scripts/python.exe
BASE_PYTHON ?= python

.PHONY: setup test lint train eval demo
setup:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Python "$(BASE_PYTHON)"
test:
	$(PYTHON) scripts/check.py
lint:
	$(PYTHON) -m ruff check .
train:
	$(PYTHON) scripts/kaggle_workflow.py submit --task phase1 $(ARGS)
eval:
	$(PYTHON) -m src.eval.rules $(ARGS)
demo:
	$(PYTHON) scripts/pending.py demo $(ARGS)
