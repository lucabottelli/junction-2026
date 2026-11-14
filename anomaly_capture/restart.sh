#!/bin/bash
# (Re)start the capture tool: stops any running copy (force-kills if it hangs), then starts one in the background.
cd "$(dirname "$0")"
PY=~/physical-ai-studio/application/backend/.venv/bin/python
for p in $(pgrep -f "^$PY -u capture.py"); do kill $p; done
for i in $(seq 20); do pgrep -f "^$PY -u capture.py" >/dev/null || break; sleep 0.5; done
for p in $(pgrep -f "^$PY -u capture.py"); do kill -9 $p; done
setsid nohup $PY -u capture.py > capture.log 2>&1 < /dev/null &
for i in $(seq 90); do curl -s -m 1 127.0.0.1:7870/status >/dev/null && break; sleep 1; done
grep -E "locked|failed|live cameras" capture.log
echo "open http://127.0.0.1:7870"
