#!/usr/bin/env bash
# Extract last JSON test result from k6 logs
docker compose -f /Users/gustavo/rinha-fraud-2026/rinha-test/docker-compose.yml logs k6 --no-color 2>&1 \
  | grep "^k6-1  | " \
  | sed 's/^k6-1  | //' \
  | awk '/^{$/{p=1; out=""} p{out = out $0 "\n"} /^}$/{p=0; print out}' \
  | tail -n 40
