import json, sys
print("%-22s %7s  " % ("run", "tok/s") + " ".join("%8s" % e for e in ["249:0","249:1","249:2","252:0","252:1","252:2","252:3"]) + "   reqs/endpoint")
for f in sys.argv[1:]:
    d = json.load(open(f)); pe = d["per_endpoint"]
    keys = sorted(pe)
    print("%-22s %7.0f  " % ("/".join(f.split("/")[-2:])[:-5], d["output_tokens_per_s"]) + " ".join("%8.3f" % pe[k]["mean_latency_s"] for k in keys) + "   " + ",".join(str(pe[k]["requests"]) for k in keys))
