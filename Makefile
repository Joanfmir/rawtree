.PHONY: up down status serve ab stop dry check monitor install abtest notebook offline
# Uso: make install · make check · make offline C="politica" · make serve POLICY=personal NAME=Leattner.v1 · make dry POLICY=personal · make monitor
POLICY ?= baseline
NAME ?= $(POLICY)
# En un worktree sin .venv propio se usa el del clon principal
PY = $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo ../rawtree/.venv/bin/python)

install:
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

check:
	$(PY) scripts/check.py

ARGS = $(if $(AB),--ab $(AB),--policy $(POLICY) --name $(NAME)) $(EXTRA)

up:      ## make up AB=... o POLICY=... NAME=...; EXTRA=--dry-run para probar
	@mkdir -p logs
	@nohup $(PY) -u scripts/supervise.py $(ARGS) >> logs/supervisor.log 2>&1 &
	@echo "Supervisor lanzado. Consultar: make status; logs/supervisor.log"

serve: up

ab: up

down:
	$(PY) scripts/supervise.py down $(EXTRA)

status:
	$(PY) scripts/supervise.py status $(EXTRA)

abtest:
	$(PY) scripts/ab.py

dry:
	$(PY) -u -m reco.serve --policy $(POLICY) --dry-run

stop: down

offline:   ## evaluación offline contrastada con el leaderboard; C="pol1 pol2" para candidatos
	$(PY) scripts/offline_vs_live.py $(if $(C),--candidates $(C)) $(EXTRA)

monitor:
	$(PY) scripts/monitor.py 30

notebook:
	$(PY) -m jupyter notebook
