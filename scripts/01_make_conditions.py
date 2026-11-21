"""Step 1: build every perturbed version of every anchor, once.

    uv run python scripts/01_make_conditions.py
    uv run python scripts/01_make_conditions.py --pairs data/generated/hatecheck_spelling.csv --name hatecheck --queries data/generated/hatecheck_tests.csv

Writes results/<name>/conditions.csv (one row per anchor x condition) and
results/<name>/queries.csv (benign / ambiguous single queries). All later
steps read these files, so both classifiers and the SLM see exactly the same texts.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import spacy

from safety_xai.config import BENIGN_FILE, CONDITIONS, PAIRS_FILE, SPACY_MODEL, run_dir
from safety_xai.data_utils import has_profanity, load_pairs, load_queries, make_conditions

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", type=Path, default=PAIRS_FILE)
parser.add_argument("--name", default="curated", help="output folder name under results/")
parser.add_argument("--queries", default=str(BENIGN_FILE), help="benign/ambiguous queries file, or 'none'")
parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
args = parser.parse_args()

out_dir = run_dir(args.name)
out_dir.mkdir(parents=True, exist_ok=True)
nlp = spacy.load(SPACY_MODEL)

pairs = load_pairs(args.pairs)
rows = []
for _, pair in pairs.iterrows():
    for c in make_conditions(pair["pair_id"], pair["anchor_phrase"], pair["perturbed_phrase"], nlp, args.conditions):
        rows.append({
            "pair_id": pair["pair_id"],
            "category": pair["category"],
            "source": pair["source"],
            "condition": c["condition"],
            "anchor": pair["anchor_phrase"],
            "text": c["text"],
            "positions": json.dumps(c["positions"]),
            "n_changed": len(c["positions"]),
            "valid": c["valid"],
            "reason": c["reason"],
            "anchor_profanity": has_profanity(pair["anchor_phrase"]),
            "text_profanity": has_profanity(c["text"]),
        })
conditions = pd.DataFrame(rows)
conditions.to_csv(out_dir / "conditions.csv", index=False)
print(f"{len(pairs)} pairs -> {len(conditions)} rows in {out_dir / 'conditions.csv'}")
print(conditions.groupby("condition")["valid"].agg(["sum", "count"]).rename(columns={"sum": "valid", "count": "total"}))

if args.queries.lower() != "none":
    queries = load_queries(args.queries)
    queries.insert(0, "query_id", range(len(queries)))
    queries["profanity"] = queries["query"].map(has_profanity)
    queries.to_csv(out_dir / "queries.csv", index=False)
    print(f"{len(queries)} single queries -> {out_dir / 'queries.csv'}")
    print(queries["query_type"].value_counts().to_string())
