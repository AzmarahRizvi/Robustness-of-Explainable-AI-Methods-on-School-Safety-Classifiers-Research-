"""Step 6 (optional): how much does each explanation method disagree with ITSELF?

    uv run python scripts/06_noise_floor.py --classifier toxic_bert --seeds 5

SHAP and LIME use random sampling, so explaining the same text twice with
different seeds gives slightly different attributions. A synonym pair only
shows real explanation instability if it changes MORE than this.
IG has no randomness; it is run twice as a sanity check (Spearman should be 1).

Output: results/<name>/<classifier>/noise_floor.csv
"""

import argparse
import itertools

import numpy as np
import pandas as pd

from safety_xai import explainers
from safety_xai.classifier import ToxicityClassifier
from safety_xai.config import CLASSIFIERS, SEED, run_dir

parser = argparse.ArgumentParser()
parser.add_argument("--classifier", choices=list(CLASSIFIERS), required=True)
parser.add_argument("--name", default="curated")
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--limit", type=int)
args = parser.parse_args()

folder = run_dir(args.name) / args.classifier
scores = pd.read_csv(folder / "scores.csv")
anchors = scores.drop_duplicates("pair_id")[["pair_id", "anchor"]]
if args.limit:
    anchors = anchors.head(args.limit)
clf = ToxicityClassifier(args.classifier)
seeds = [SEED + i for i in range(args.seeds)]

rows = []
for i, (pid, text) in enumerate(zip(anchors["pair_id"], anchors["anchor"]), 1):
    runs = {
        "ig": [explainers.integrated_gradients(clf, text)[0] for _ in range(2)],
        "shap": [explainers.shap_values(clf, text, seed=s) for s in seeds],
        "lime": [explainers.lime_weights(clf, text, seed=s) for s in seeds],
    }
    for method, vectors in runs.items():
        # compare every pair of runs, then average
        comps = [explainers.compare(a, b, []) for a, b in itertools.combinations(vectors, 2)]
        row = {"pair_id": pid, "method": method, "runs": len(vectors)}
        for k in ["pearson", "spearman", "top10_overlap", "top30_overlap", "top50_overlap"]:
            row[k] = float(np.nanmean([c[k] for c in comps]))
        rows.append(row)
    print(f"[{i}/{len(anchors)}] pair {pid}: " + ", ".join(f"{r['method']} {r['spearman']:.3f}" for r in rows[-3:]))

out = pd.DataFrame(rows)
out.to_csv(folder / "noise_floor.csv", index=False)
print(out.groupby("method")[["spearman", "top30_overlap"]].mean().round(3))
