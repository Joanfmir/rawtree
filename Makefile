# Uso: make install · make check · make serve POLICY=personal NAME=Leattner.v1 · make dry POLICY=personal · make monitor
POLICY ?= baseline
NAME ?= $(POLICY)
# Test A/B: make serve POLICY=personal AB=hgb AB_SHARE=0.2   (el 20 % de las peticiones va a hgb)
AB ?=
AB_SHARE ?= 0.5
ABFLAGS = $(if $(AB),--ab $(AB) --ab-share $(AB_SHARE))
PY = .venv/bin/python

install:
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

check:
	$(PY) scripts/check.py

serve:   ## producción: nohup + caffeinate para que el Mac no se duerma
	caffeinate -dimsu nohup $(PY) -u -m reco.serve --policy $(POLICY) --name $(NAME) $(ABFLAGS) >> logs/serve.log 2>&1 &
	@echo "Corriendo en segundo plano. Logs: tail -f logs/serve.log"

dry:
	$(PY) -u -m reco.serve --policy $(POLICY) $(ABFLAGS) --dry-run

stop:
	pkill -f "reco.serve" || true

train:   ## entrena el re-ranker hgb para producción (data/models/hgb_live.pkl)
	$(PY) scripts/train_hgb.py

monitor:
	$(PY) scripts/monitor.py 30

notebook:
	$(PY) -m jupyter notebook
