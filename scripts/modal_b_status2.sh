#!/bin/bash
# Status of the two 159-episode SmolVLA runs launched 2026-11-15 04:52 on the second Modal account (A / B = different validation sets)
cd ~/physical-ai-studio/application/backend
export MODAL_PROFILE=andreacevenini-aldini
for j in f52ed47a-46a2-41c7-b7e3-71a38be69bfc 4c9a6f02-22a1-460d-a765-77d6cc176fcb; do
  rm -f /tmp/st_$j.json; .venv/bin/modal volume get physicalai-training-b jobs/$j/status.json /tmp/st_$j.json >/dev/null 2>&1
  echo "$j: $(cut -c1-300 /tmp/st_$j.json 2>/dev/null || echo 'no status yet')"
done
