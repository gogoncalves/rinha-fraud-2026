#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/rinha-test"
rm -f test/results.json
docker compose --profile test up --abort-on-container-exit >/dev/null 2>&1
cat test/results.json | python3 -m json.tool
