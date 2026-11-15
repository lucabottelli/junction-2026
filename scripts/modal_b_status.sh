#!/bin/bash
cd ~/physical-ai-studio/application/backend
export MODAL_PROFILE=andreacevenini-aldini
for j in e11bf6a0-88ff-4218-8cf2-d7e937b4d72a 17ac5059-c5e6-408a-9b7d-0309aecae2a5; do
  rm -f /tmp/st_$j.json; .venv/bin/modal volume get physicalai-training-b jobs/$j/status.json /tmp/st_$j.json >/dev/null 2>&1
  echo "$j: $(cut -c1-300 /tmp/st_$j.json 2>/dev/null || echo 'no status yet')"
done
.venv/bin/modal container list --env main 2>&1 | grep -E 'ta-|No ' | cut -c1-150
