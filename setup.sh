#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
conda run -n soccertwos_x86 python example_random_players.py
