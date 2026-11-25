"""Exact-token gate for run_ab.sh: compare every gate-<arm>/gate.json against the first arm's.

Each gate cell is a deterministic cold start running the same fixed workload (seed, prompts, 128 requests,
--ignore-eos, 128 new tokens), so with batch-invariant deterministic inference every output token must be
identical across arms regardless of batching. Exit 1 if any arm differs."""
import json, os, sys
run = sys.argv[1]
plan = open(os.path.join(run, "plan.sh")).read()
order = []
for line in plan.splitlines():
    if line.strip().startswith("ARMS=("):
        order = line.split("(", 1)[1].split(")")[0].split()
for line in plan.splitlines():
    # GATE_ARMS="a b" (run_ab.sh) restricts the gate cells; its first arm is the reference.
    if line.strip().startswith("GATE_ARMS="):
        order = line.split("=", 1)[1].strip().strip('"').strip("'").split()
gates = {a: os.path.join(run, "gate-" + a, "gate.json") for a in order}
missing = [a for a, p in gates.items() if not os.path.exists(p)]
if missing:
    print("GATE incomplete: no gate.json for", missing); sys.exit(1)
def load(p):
    d = json.load(open(p))
    return {r["index"]: (r.get("prompt_index"), r.get("output_token_ids") or []) for r in d["records"] if r.get("success")}, d
ref_arm = order[0]; ref, _ = load(gates[ref_arm])
ok = True
for a in order[1:]:
    cur, _ = load(gates[a])
    common = sorted(set(ref) & set(cur))
    tok_total = tok_same = req_same = 0; first_bad = None
    for i in common:
        pa, ta = ref[i]; pb, tb = cur[i]
        if pa != pb: first_bad = first_bad or (i, "prompt mismatch"); continue
        n = max(len(ta), len(tb)); same = sum(1 for x, y in zip(ta, tb) if x == y)
        tok_total += n; tok_same += same; req_same += (ta == tb)
        if ta != tb and first_bad is None:
            k = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            first_bad = (i, "first differing token at position %d" % k)
    exact = len(common) > 0 and req_same == len(common) == len(ref) == len(cur)
    ok &= exact
    print("  GATE %s vs %s: requests %d/%d identical, tokens %.4f%% identical%s -> %s" % (
        a, ref_arm, req_same, len(ref), 100.0 * tok_same / max(1, tok_total),
        "" if first_bad is None else ", first mismatch: request %s (%s)" % first_bad, "PASS" if exact else "FAIL"))
sys.exit(0 if ok else 1)
