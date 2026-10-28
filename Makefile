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
	$(PYTHON) scripts/pending.py train $(ARGS)
eval:
	$(PYTHON) scripts/pending.py eval $(ARGS)
demo:
	$(PYTHON) scripts/pending.py demo $(ARGS)
