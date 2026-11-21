#!/bin/bash
set -euo pipefail

W=${DT_EXPERIMENT_WORKSPACE:-/data/lishuhuan/dt-v3-pilot}
SOURCE=${DT_EXPERIMENT_SOURCE:-/data/lishuhuan/dt-v3-page}
stamp=${1:?stamp required}
label=${2:?label required}
out="$W/target-combo-results/$stamp/$label/benchmark.json"
topology=${DT_COMBO_TOPOLOGY_FILE:-$W/topology-7t2d-page-ab.toml}

mkdir -p "$(dirname "$out")"

mapfile -t endpoints < <("$SOURCE/python-local" - "$topology" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    topology = tomllib.load(f)
hosts = topology["hosts"]
for instance in topology["instances"]:
    if instance["role"] != "target":
        continue
    addr = hosts[instance["host"]]["addr"]
    print(f"http://{addr}:{instance['http_port']}")
PY
)
endpoint_args=()
for endpoint in "${endpoints[@]}"; do
  endpoint_args+=(--endpoint "$endpoint")
done

exec "$SOURCE/python-local" -m dt_benchmark \
  "${endpoint_args[@]}" \
  --prompts-file "$W/prompts.json" --output "$out" \
  --global-concurrency 48 --routing-policy least-inflight \
  --warmup-requests 64 --measurement-requests 384 \
  --max-new-tokens 128 --ignore-eos --seed 20261121 \
  --health-timeout-s 900 --request-timeout-s 180 --e2e-slo-ms 10000
