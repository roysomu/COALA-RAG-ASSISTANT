PYTHON ?= python3.12
VENV := .venv/bin
CURRENT_THREAD ?=

.PHONY: setup check init-db ingest-samples run test storage-report prune-dry-run prune
setup:
	@test -x $(VENV)/python || $(PYTHON) -m venv .venv
	$(VENV)/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 .venv"'
	$(VENV)/python -m pip install --no-cache-dir --upgrade pip
	$(VENV)/python -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
check:
	$(VENV)/python scripts/check_setup.py
init-db:
	$(VENV)/python scripts/init_db.py
ingest-samples:
	$(VENV)/python scripts/ingest_samples.py
run:
	$(VENV)/streamlit run app.py
test:
	$(VENV)/python -m pytest -q
storage-report:
	$(VENV)/python scripts/storage_report.py
prune-dry-run:
	$(VENV)/python scripts/prune_checkpoints.py --dry-run $(if $(CURRENT_THREAD),--current-thread $(CURRENT_THREAD),)
prune:
	$(VENV)/python scripts/prune_checkpoints.py --apply $(if $(CURRENT_THREAD),--current-thread $(CURRENT_THREAD),)
