PYTHON ?= .venv/bin/python
export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export OMP_NUM_THREADS := 2
DIFFICULTY_WORKERS ?= 4
SB3_WORKERS ?= 4
FRESH_TRANSFER_WORKERS ?= 4
LAYOUT_TRANSFER_WORKERS ?= 8
.PHONY: test q neural playtest ablations curve curve-figure unseen-protocol unseen difficulty-protocol difficulty difficulty-figure bomb-dqn-longer bomb-matched-longer fresh-transfer sb3-protocol sb3 defect-protocol defect reward-challenge reward-oracle gridquest-transfer figures all
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
bomb-dqn-longer:
	$(PYTHON) -m analysis.bomb_dqn_longer
bomb-matched-longer:
	$(PYTHON) -m analysis.bomb_matched_longer
fresh-transfer:
	$(PYTHON) -m analysis.fresh_transfer --workers $(FRESH_TRANSFER_WORKERS)
sb3-protocol:
	$(PYTHON) -m analysis.sb3_reference --write-protocol
sb3:
	$(PYTHON) -m analysis.sb3_reference --run --workers $(SB3_WORKERS)
defect-protocol:
	$(PYTHON) -m playtest.new_defect --write-protocol
defect:
	$(PYTHON) -m playtest.new_defect --run
reward-challenge:
	$(PYTHON) -m playtest.frozen_oracle_reward
reward-oracle:
	$(PYTHON) -m playtest.reward_oracle_run
gridquest-transfer:
	$(PYTHON) -m analysis.gridquest_transfer --workers $(LAYOUT_TRANSFER_WORKERS)
figures:
	$(PYTHON) -m analysis.figures
