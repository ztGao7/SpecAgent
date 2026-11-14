"""Score and compare VStarBench runs: python compare_vstar.py baseline=<work_dir> async=<work_dir> [label=<work_dir> ...]
The first run is the reference for speed-up and paired sign tests. Reads the prediction xlsx and *_sample_wall_clock.csv."""
import sys, glob, re, math, statistics as st, collections, pandas as pd
def norm(t): return re.sub(r"\s+", " ", str(t).strip().lower()).strip(" .()")
def ok(a, g, q):
    if a is None or str(a) == "nan": return False
    a, g = norm(a), norm(g)
    if a == g: return True
    m = re.fullmatch(r"([a-h])", a)
    if m:
        opts = dict(re.findall(r"(?m)^\s*\(?([A-H])[.)]\s+([^\n]+)", q or ""))
        if norm(opts.get(m.group(1).upper(), "\0")) == g: return True
    return False
ROUTE = {"fast_main_answer": "fast", "main_answer_draft_evidence_verified": "verified",
         "main_answer_from_draft_evidence": "early", "main_answer_fallback": "fallback"}
def load(d):
    xs = [x for x in glob.glob(d + "/**/*.xlsx", recursive=True) if not re.search(r"_(acc|score|rating|openai|gpt)", x)]
    df = pd.read_excel(sorted(xs)[-1])
    wc = glob.glob(d + "/**/*_sample_wall_clock.csv", recursive=True)
    W = {str(r["index"]): float(r["sample_wall_clock_s"]) for _, r in pd.read_csv(sorted(wc)[-1]).iterrows()}
    out = {}
    for _, r in df.iterrows():
        q = str(r.get("question", "")) + "\n" + "\n".join(f"{L}. {r[L]}" for L in "ABCDEFGH" if L in r and isinstance(r[L], str))
        s = re.search(r"stop_reason.: .([a-z_]+)", str(r.get("extra_records", ""))); sr = s.group(1) if s else ""
        out[str(r["index"])] = dict(ok=ok(r["prediction"], r["answer"], q), wall=W[str(r["index"])],
                                    route=ROUTE.get(sr, sr or "-"), cat=str(r.get("category", "")), pred=str(r["prediction"]))
    return out
def sign(x, y):
    n = x + y; k = min(x, y); return min(1, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2) if n else 1
runs = [a.split("=", 1) for a in sys.argv[1:]]
L = {k: load(v) for k, v in runs}; ref = runs[0][0]; b = L[ref]
us = sorted(set.intersection(*[set(x) for x in L.values()]))
bm, bmd = st.mean(b[u]["wall"] for u in us), st.median(b[u]["wall"] for u in us)
print(f"n={len(us)} common items; reference = {ref}")
for k, r in L.items():
    w = [r[u]["wall"] for u in us]; R = sum(r[u]["ok"] for u in us)
    x = sum(b[u]["ok"] and not r[u]["ok"] for u in us); y = sum(r[u]["ok"] and not b[u]["ok"] for u in us)
    print(f"{k:16s} right {R} ({100*R/len(us):.2f}%) mean {st.mean(w):.1f}s median {st.median(w):.1f}s "
          f"speedup {bm/st.mean(w):.2f}x/{bmd/st.median(w):.2f}x | {ref}-only:{k}-only {x}:{y} p={sign(x,y):.2f}")
    g = collections.defaultdict(list)
    for u in us: g[r[u]["route"]].append(u)
    if len(g) > 1:
        print("   routes:", " | ".join(f"{q} {len(v)} (right {sum(r[u]['ok'] for u in v)}, {st.mean(r[u]['wall'] for u in v):.0f}s)"
                                    for q, v in sorted(g.items(), key=lambda kv: -len(kv[1]))),
              "| 'error' outputs:", sum(r[u]["pred"].strip().lower() == "error" for u in us))
for c in sorted(set(b[u]["cat"] for u in us)):
    cu = [u for u in us if b[u]["cat"] == c]
    print(f"  {c:20s} n={len(cu)} " + " | ".join(f"{k} {sum(r[u]['ok'] for u in cu)} {st.mean(r[u]['wall'] for u in cu):.0f}s" for k, r in L.items()))
