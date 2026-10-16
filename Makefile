PYTHON ?= .venv/bin/python
export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export OMP_NUM_THREADS := 2
.PHONY: test q neural playtest ablations figures all
all: q neural playtest ablations figures
test:
	$(PYTHON) -m pytest -q
q:
	$(PYTHON) -m experiments.run --suite q --seeds 5 --episodes 500
neural:
	$(PYTHON) -m experiments.run --suite neural --seeds 5 --steps 5120
playtest:
	$(PYTHON) -m experiments.run --suite playtest --seeds 5 --max-hunt-steps 1200
ablations:
	$(PYTHON) -m experiments.run --suite ablations --seeds 5 --steps 5120
figures:
	$(PYTHON) -m analysis.figures
