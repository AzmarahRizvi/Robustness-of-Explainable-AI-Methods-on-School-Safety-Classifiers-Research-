"""Step 3: run the local SLM (Llama-3-8B via Ollama) on EVERY text.

    ollama pull llama3:8b
    uv run python scripts/03_run_slm.py

The SLM never sees the classifier or its score. It runs on all anchors, all
perturbed versions and all benign / ambiguous queries, so it can catch risks
the classifier misses and we can count its false alarms.

Output in results/<name>/:
    slm.csv         one row per unique text: intent sentence, risk, category
    slm_pairs.csv   one row per anchor x condition: anchor vs perturbed narrative
                    (graph similarity and embedding similarity)
    slm_speed.json  seconds per query and memory used, on this machine

Answers are cached in results/llm_cache.jsonl, so stopping and restarting is safe.
"""

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import psutil
import spacy

from safety_xai import llm, narratives
from safety_xai.config import EMBEDDING_MODEL, LLM_OPTIONS, SLM_MODEL, SPACY_MODEL, run_dir

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="curated")
parser.add_argument("--model", default=SLM_MODEL)
parser.add_argument("--limit", type=int, help="only the first N texts (quick test)")
parser.add_argument("--queries-per-type", type=int,
                    help="only this many single queries per query_type (the SLM is slow on a CPU)")
args = parser.parse_args()
llm.require_models([args.model])

folder = run_dir(args.name)
conditions = pd.read_csv(folder / "conditions.csv")
conditions = conditions[conditions["valid"]]

# Order: anchors, then synonym versions, then controls, then single queries.
# If you stop early, the most important texts are already done.
texts = list(conditions["anchor"].drop_duplicates())
for cond in conditions["condition"].unique():
    texts += list(conditions.loc[conditions["condition"] == cond, "text"])
queries = None
if (folder / "queries.csv").exists():
    queries = pd.read_csv(folder / "queries.csv")
    if args.queries_per_type:
        queries = queries.groupby("query_type", group_keys=False).head(args.queries_per_type)
    texts += list(queries["query"].astype(str))
texts = list(dict.fromkeys(texts))
if args.limit:
    texts = texts[:args.limit]

rows = []
start = time.perf_counter()
for i, text in enumerate(texts, 1):
    r = llm.analyse_query(text, args.model)
    ans = r["answer"] or {}
    rows.append({"text": text, "intent": ans.get("intent"), "risk": ans.get("risk"), "category": ans.get("category"),
                 "ok": r["ok"], "error": r["error"], "seconds": r["seconds"], "cached": r["cached"]})
    if not r["cached"]:
        print(f"[{i}/{len(texts)}] {ans.get('risk')}/{ans.get('category')} ({r['seconds'] or 0:.0f}s) {text[:60]}")
    if not r["ok"]:
        print(f"   FAILED: {r['error']}")
memory = llm.ollama_memory()  # read while the model is still loaded
slm = pd.DataFrame(rows)
slm.to_csv(folder / "slm.csv", index=False)
print(f"{len(slm)} texts, {int((~slm['ok']).sum())} failed, {int((~slm['cached']).sum())} new calls")

# ---------------------------------------------------------------- speed and memory
new = slm.loc[slm["ok"] & ~slm["cached"], "seconds"]
all_ok = slm.loc[slm["ok"], "seconds"]
speed = {
    "model": args.model, "options": LLM_OPTIONS,
    "seconds_per_query_new_calls": {"n": len(new), "mean": new.mean(), "median": new.median(),
                                    "p95": new.quantile(0.95)} if len(new) else None,
    "seconds_per_query_all_answers": {"n": len(all_ok), "mean": all_ok.mean(), "median": all_ok.median(),
                                             "p95": all_ok.quantile(0.95)},
    "ollama_loaded_models": memory,
    "machine": {"platform": platform.platform(), "cpu": platform.processor(), "cores": psutil.cpu_count(),
                "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)},
    "wall_seconds_this_session": time.perf_counter() - start,
}
(folder / "slm_speed.json").write_text(json.dumps(speed, indent=2, default=float), encoding="utf-8")

# ---------------------------------------------------------------- anchor vs perturbed narratives
by_text = slm.set_index("text")
nlp = spacy.load(SPACY_MODEL)
embedder = narratives.Embedder(EMBEDDING_MODEL)
pair_rows = []
for _, c in conditions.iterrows():
    if c["anchor"] not in by_text.index or c["text"] not in by_text.index:
        continue
    a, p = by_text.loc[c["anchor"]], by_text.loc[c["text"]]
    row = {"pair_id": c["pair_id"], "category": c["category"], "condition": c["condition"],
           "anchor": c["anchor"], "text": c["text"],
           "anchor_risk": a["risk"], "anchor_slm_category": a["category"], "anchor_intent": a["intent"],
           "text_risk": p["risk"], "text_slm_category": p["category"], "text_intent": p["intent"]}
    if a["ok"] and p["ok"]:
        sim, finished = narratives.graph_similarity(narratives.make_graph(nlp, a["intent"]),
                                                    narratives.make_graph(nlp, p["intent"]))
        row["graph_similarity"], row["ged_finished"] = sim, finished
        row["embedding_similarity"] = float(embedder.cosine([a["intent"]], [p["intent"]])[0])
    pair_rows.append(row)
pd.DataFrame(pair_rows).to_csv(folder / "slm_pairs.csv", index=False)
print(f"Wrote {folder / 'slm.csv'}, slm_pairs.csv and slm_speed.json")
if len(new):
    print(f"Speed: {np.mean(new):.1f} s per query (mean of {len(new)} new calls). Memory: {memory}")
