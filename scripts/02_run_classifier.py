"""Step 2: score every text with one classifier and explain it with IG, SHAP and LIME.

    uv run python scripts/02_run_classifier.py --classifier toxic_bert
    uv run python scripts/02_run_classifier.py --classifier toxic_roberta
    uv run python scripts/02_run_classifier.py --classifier toxic_bert --name hatecheck --no-explain

Output in results/<name>/<classifier>/:
    scores.csv          one row per anchor x condition: scores, alert decisions,
                        sub-word counts, and explanation-stability numbers
    attributions.jsonl  the full word-level attribution vectors
    queries_scores.csv  scores for the benign / ambiguous queries
    run_info.json       model commit, library versions, hardware, timings

The run can be stopped and restarted: finished pairs are skipped
(use --restart to start from zero). On a CPU, explanations take a few hours.
"""

import argparse
import json
import platform
import time
from importlib import metadata

import pandas as pd
import psutil
import torch

from safety_xai import explainers
from safety_xai.classifier import ToxicityClassifier
from safety_xai.config import CLASSIFIERS, IG_STEPS, LIME_SAMPLES, SEED, SHAP_EVALS, THRESHOLD, TOP_K, run_dir

parser = argparse.ArgumentParser()
parser.add_argument("--classifier", choices=list(CLASSIFIERS), required=True)
parser.add_argument("--name", default="curated")
parser.add_argument("--threshold", type=float, default=THRESHOLD)
parser.add_argument("--no-explain", action="store_true", help="scores only (fast; for big public sets)")
parser.add_argument("--restart", action="store_true", help="delete earlier output and start again")
parser.add_argument("--limit", type=int, help="only the first N pairs (for a quick test)")
args = parser.parse_args()

in_dir = run_dir(args.name)
out_dir = in_dir / args.classifier
out_dir.mkdir(parents=True, exist_ok=True)
scores_file = out_dir / "scores.csv"
attr_file = out_dir / "attributions.jsonl"
if args.restart:
    scores_file.unlink(missing_ok=True)
    attr_file.unlink(missing_ok=True)

conditions = pd.read_csv(in_dir / "conditions.csv")
conditions = conditions[conditions["valid"]]
pair_ids = list(dict.fromkeys(conditions["pair_id"]))
if args.limit:
    pair_ids = pair_ids[:args.limit]
done = set(pd.read_csv(scores_file)["pair_id"]) if scores_file.exists() else set()
print(f"{len(pair_ids)} pairs, {len(done)} already done")

start = time.perf_counter()
clf = ToxicityClassifier(args.classifier, args.threshold)
load_seconds = time.perf_counter() - start


def score_columns(prefix, s):
    return {f"{prefix}_score": s["score"], f"{prefix}_flagged": s["flagged"],
            f"{prefix}_max_harm_score": s["max_harm_score"], f"{prefix}_max_harm_label": s["max_harm_label"]}


# ---------------------------------------------------------------- pairs
for n, pid in enumerate(pair_ids, 1):
    if pid in done:
        continue
    t0 = time.perf_counter()
    group = conditions[conditions["pair_id"] == pid]
    anchor = group["anchor"].iloc[0]
    a = clf.score_one(anchor)
    ex_a = None if args.no_explain else explainers.explain(clf, anchor)
    # The risk words are the words the synonym version swapped. Every control has
    # the same number of changed words, so the synonym row is always there.
    synonym_row = group[group["condition"] == "synonym"]
    risk_positions = json.loads(synonym_row["positions"].iloc[0])

    rows, attr_lines = [], []
    for _, c in group.iterrows():
        text = c["text"]
        p = clf.score_one(text)
        row = {"pair_id": pid, "category": c["category"], "condition": c["condition"], "anchor": anchor, "text": text,
               "positions": c["positions"], "anchor_profanity": c["anchor_profanity"],
               "text_profanity": c["text_profanity"], "label": clf.label, "threshold": clf.threshold,
               **score_columns("anchor", a), **score_columns("text", p),
               "delta": p["score"] - a["score"],
               "evaded": a["flagged"] and not p["flagged"],
               "newly_flagged": (not a["flagged"]) and p["flagged"],
               "alert_flipped": a["flagged"] != p["flagged"],
               "anchor_subwords": clf.n_subwords(anchor), "text_subwords": clf.n_subwords(text)}
        row["extra_subwords"] = row["text_subwords"] - row["anchor_subwords"]

        if ex_a is not None:
            ex_p = explainers.explain(clf, text)
            for m in explainers.METHODS:
                for k, v in explainers.compare(ex_a[m], ex_p[m], risk_positions).items():
                    row[f"{m}_{k}"] = v
            for side, ex in (("anchor", ex_a), ("text", ex_p)):
                chk = ex["ig_check"]
                row[f"{side}_ig_attr_sum"] = chk["attr_sum"]
                row[f"{side}_ig_f_diff"] = chk["f_input"] - chk["f_baseline"]
                row[f"{side}_ig_delta"] = chk["convergence_delta"]
            attr_lines.append(json.dumps({"pair_id": int(pid), "condition": c["condition"],
                                          "anchor_words": anchor.split(), "text_words": text.split(),
                                          **{f"anchor_{m}": ex_a[m].tolist() for m in explainers.METHODS},
                                          **{f"text_{m}": ex_p[m].tolist() for m in explainers.METHODS}}))
        rows.append(row)

    # save after every pair, so a stopped run can continue
    pd.DataFrame(rows).to_csv(scores_file, mode="a", header=not scores_file.exists(), index=False)
    if attr_lines:
        with attr_file.open("a", encoding="utf-8") as f:
            f.write("\n".join(attr_lines) + "\n")
    print(f"[{n}/{len(pair_ids)}] pair {pid}: anchor {a['score']:.3f} | "
          + ", ".join(f"{r['condition']} {r['text_score']:.3f}" for r in rows) + f" ({time.perf_counter() - t0:.0f}s)")
pairs_seconds = time.perf_counter() - start - load_seconds

# ---------------------------------------------------------------- single queries
queries_file = in_dir / "queries.csv"
if queries_file.exists():
    q = pd.read_csv(queries_file)
    t0 = time.perf_counter()
    scores = clf.score(q["query"].astype(str).tolist())
    q["score"] = scores
    q["flagged"] = scores >= clf.threshold
    q.to_csv(out_dir / "queries_scores.csv", index=False)
    per_query = (time.perf_counter() - t0) / max(1, len(q))
    print(f"Scored {len(q)} single queries. Flag rate by type:\n{q.groupby('query_type')['flagged'].mean()}")
else:
    per_query = None


# ---------------------------------------------------------------- run info
def version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


info = {
    "classifier": args.classifier, "hf_name": CLASSIFIERS[args.classifier]["hf_name"], "hf_commit": clf.commit,
    "label": clf.label, "threshold": clf.threshold, "explained": not args.no_explain,
    "settings": {"seed": SEED, "ig_steps": IG_STEPS, "ig_output": "sigmoid probability", "ig_baseline": "PAD tokens",
                 "shap_max_evals": SHAP_EVALS, "lime_samples": LIME_SAMPLES, "top_k": TOP_K},
    "hardware": {"platform": platform.platform(), "cpu": platform.processor(), "cores": psutil.cpu_count(),
                 "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "cuda": torch.cuda.is_available(),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    "seconds": {"load_model": load_seconds, "pairs_this_session": pairs_seconds,
                "score_one_query": per_query},
    "process_ram_gb_at_end": round(psutil.Process().memory_info().rss / 1e9, 2),
    "versions": {p: version(p) for p in ["torch", "transformers", "captum", "shap", "lime", "spacy", "nltk",
                                         "scipy", "statsmodels", "pandas", "numpy"]},
}
(out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(f"Done. Results in {out_dir}")
