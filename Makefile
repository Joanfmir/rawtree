# Uso: make install · make check · make serve POLICY=personal NAME=Leattner.v1 · make dry POLICY=personal · make monitor
POLICY ?= baseline
NAME ?= $(POLICY)
PY = .venv/bin/python

install:
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

check:
	$(PY) scripts/check.py

serve:   ## producción: nohup + caffeinate para que el Mac no se duerma
	caffeinate -dimsu nohup $(PY) -u -m reco.serve --policy $(POLICY) --name $(NAME) >> logs/serve.log 2>&1 &
	@echo "Corriendo en segundo plano. Logs: tail -f logs/serve.log"

dry:
	$(PY) -u -m reco.serve --policy $(POLICY) --dry-run

stop:
	pkill -f "reco.serve" || true

monitor:
	$(PY) scripts/monitor.py 30

notebook:
	$(PY) -m jupyter notebook
