PYTHON ?= .venv/bin/python
export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export OMP_NUM_THREADS := 2
DIFFICULTY_WORKERS ?= 4
SB3_WORKERS ?= 4
.PHONY: test q neural playtest ablations curve curve-figure unseen-protocol unseen difficulty-protocol difficulty difficulty-figure sb3-protocol sb3 defect-protocol defect figures all
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
curve:
	$(PYTHON) -m experiments.run --suite curve --seeds 5
curve-figure:
	$(PYTHON) -m analysis.figures --curve evidence/learning_curve_20480.json
unseen-protocol:
	$(PYTHON) -m playtest.unseen --write-protocol
unseen:
	$(PYTHON) -m playtest.unseen --run
difficulty-protocol:
	$(PYTHON) -m analysis.difficulty --write-protocol
difficulty:
	$(PYTHON) -m analysis.difficulty --run --workers $(DIFFICULTY_WORKERS)
difficulty-figure:
	$(PYTHON) -m analysis.figures --difficulty evidence/slicer_difficulty_20480.json
sb3-protocol:
	$(PYTHON) -m analysis.sb3_reference --write-protocol
sb3:
	$(PYTHON) -m analysis.sb3_reference --run --workers $(SB3_WORKERS)
defect-protocol:
	$(PYTHON) -m playtest.new_defect --write-protocol
defect:
	$(PYTHON) -m playtest.new_defect --run
figures:
	$(PYTHON) -m analysis.figures
