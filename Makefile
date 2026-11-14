# Dev targets (no hardware needed). See docs/DEV.md.
PY ?= .venv/bin/python
PORT ?= 7871

.PHONY: venv mock test test-flows test-ui test-all simtest

venv:            ## create .venv from requirements-dev.txt
	python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements-dev.txt

mock:            ## run the control server on the mock station: http://127.0.0.1:$(PORT)/  (data in mock_data/)
	JUNCTION_MOCK=1 JUNCTION_PORT=$(PORT) $(PY) -u anomaly_capture/capture.py

test:            ## fast tests (unit, API, sim_test, other agents' tests): ~1-2 min
	$(PY) -m pytest tests -q -m "not slow and not ui"

test-flows:      ## end-to-end flows on the mock (collect, sort, STOP/recovery): ~8 min
	$(PY) -m pytest tests -q -m slow

test-ui:         ## browser smoke test (Node + Playwright + Chromium), screenshots in tests/screenshots/
	$(PY) -m pytest tests -q -m ui -s

test-all:
	$(PY) -m pytest tests -q

simtest:
	$(PY) anomaly_capture/sim_test.py
