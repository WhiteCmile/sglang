#!/bin/bash
set -euo pipefail

W=${DT_EXPERIMENT_WORKSPACE:-/data/lishuhuan/dt-v3-pilot}
SOURCE=${DT_EXPERIMENT_SOURCE:-/data/lishuhuan/dt-v3-page}
stamp=${1:?stamp required}
label=${2:?label required}
out="$W/target-combo-results/$stamp/$label/benchmark.json"

mkdir -p "$(dirname "$out")"

exec "$SOURCE/python-local" -m dt_benchmark \
  --endpoint http://192.168.1.249:38100 \
  --endpoint http://192.168.1.249:38101 \
  --endpoint http://192.168.1.249:38102 \
  --endpoint http://192.168.1.252:38100 \
  --endpoint http://192.168.1.252:38101 \
  --endpoint http://192.168.1.252:38102 \
  --endpoint http://192.168.1.252:38103 \
  --prompts-file "$W/prompts.json" --output "$out" \
  --global-concurrency 48 --routing-policy least-inflight \
  --warmup-requests 64 --measurement-requests 384 \
  --max-new-tokens 128 --ignore-eos --seed 20261121 \
  --health-timeout-s 900 --request-timeout-s 180 --e2e-slo-ms 10000
