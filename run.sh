#!/usr/bin/env bash
# One-shot replicate: create a venv, install deps, run the probe.
# The RT-J checkpoints (~350MB) and the MiniLM encoder download on first run
# and are cached under ~/.cache/huggingface for subsequent runs.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo ">> creating .venv"
  "$PYTHON" -m venv .venv
fi

echo ">> installing deps"
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

echo ">> running sentiment probe"
./.venv/bin/python sentiment.py
