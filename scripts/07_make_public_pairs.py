"""Step 7 (scale-up): make more pairs from public data and standard attacks.

    # HateCheck (Roettger et al. 2021): human-validated spelling-variant pairs + labelled single tests
    uv run python scripts/07_make_public_pairs.py hatecheck

    # Non-offensive tweets, as a bigger benign set for false alarms
    uv run python scripts/07_make_public_pairs.py tweets --n 300

    # TextFooler / BAE baselines through TextAttack (needs: uv sync --extra attacks)
    uv run python scripts/07_make_public_pairs.py attack --recipe textfooler --classifier toxic_bert --source curated
    uv run python scripts/07_make_public_pairs.py attack --recipe bae --classifier toxic_bert --source tweet_eval --n 300

Every output is a pairs CSV in data/generated/ that script 01 reads, e.g.
    uv run python scripts/01_make_conditions.py --pairs data/generated/curated_textfooler_toxic_bert.csv --name textfooler_bert --queries none

Attacks only start from TRUE POSITIVES (texts the classifier flags) and count as
successful only when the score drops below the threshold. TextFooler and BAE
use the filters the reviewers mention: counter-fitted embeddings or BERT-MLM
candidates, same part of speech, USE sentence similarity, no stop-word changes,
and a limit on how many words change.

Note on Jigsaw: both classifiers were trained on Jigsaw data, and the Jigsaw test
labels are only on Kaggle (manual download). We use tweet_eval and HateCheck,
which neither model was trained on.
"""

import argparse
import json
import random

import numpy as np
import pandas as pd

from safety_xai.classifier import ToxicityClassifier
from safety_xai.config import CLASSIFIERS, DATA_DIR, PAIRS_FILE, SEED, THRESHOLD
from safety_xai.data_utils import check_pair, load_pairs

OUT = DATA_DIR / "generated"
# spelling variants that keep the number of words, so positions still line up
SPELLING_TYPES = ["spell_char_swap_h", "spell_char_del_h", "spell_leet_h"]


def load_hatecheck():
    from datasets import load_dataset

    df = load_dataset("Paul/hatecheck", split="test").to_pandas()
    df["test_case"] = df["test_case"].map(lambda t: " ".join(str(t).split()))
    return df


def make_hatecheck(per_functionality):
    hc = load_hatecheck()
    # pairs: original hateful case -> its spelling variant
    original = hc.set_index("case_id")["test_case"]
    var = hc[hc["functionality"].isin(SPELLING_TYPES) & hc["ref_case_id"].notna()].copy()
    var["anchor_phrase"] = var["ref_case_id"].astype(int).map(original)
    var = var[var["anchor_phrase"].notna()]
    var["ok"] = [check_pair(a, p)[0] for a, p in zip(var["anchor_phrase"], var["test_case"])]
    var = var[var["ok"]].reset_index(drop=True)
    pairs = pd.DataFrame({"pair_id": range(len(var)), "category": "hatecheck_" + var["functionality"],
                          "anchor_phrase": var["anchor_phrase"], "perturbed_phrase": var["test_case"],
                          "source": "hatecheck"})
    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT / "hatecheck_spelling.csv", index=False)
    print(f"{len(pairs)} HateCheck spelling pairs -> {OUT / 'hatecheck_spelling.csv'}")

    # single tests: e.g. profanity_nh (swearing, not hateful) vs derog_impl_h (implicit hate, no swearing)
    tests = hc[~hc["functionality"].str.startswith("spell_")]
    tests = tests.sample(frac=1, random_state=SEED).groupby("functionality").head(per_functionality)
    q = pd.DataFrame({"query": tests["test_case"],
                      "query_type": np.where(tests["label_gold"] == "hateful", "harmful", "benign"),
                      "topic": tests["functionality"]})
    q.to_csv(OUT / "hatecheck_tests.csv", index=False)
    print(f"{len(q)} HateCheck single tests ({per_functionality} per functionality) -> {OUT / 'hatecheck_tests.csv'}")


def make_tweets(n):
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "offensive", split="test")
    texts = [" ".join(r["text"].split()) for r in ds if r["label"] == 0]
    texts = [t for t in texts if 3 <= len(t.split()) <= 30]
    random.Random(SEED).shuffle(texts)
    q = pd.DataFrame({"query": texts[:n], "query_type": "benign", "topic": "tweet_eval_not_offensive"})
    OUT.mkdir(parents=True, exist_ok=True)
    q.to_csv(OUT / "tweet_benign.csv", index=False)
    print(f"{len(q)} benign tweets -> {OUT / 'tweet_benign.csv'}")


# ---------------------------------------------------------------- TextAttack
def harmful_texts(source, n_candidates):
    """(text, category) of harmful items. Test splits only."""
    if source == "curated":
        p = load_pairs(PAIRS_FILE)
        return list(zip(p["anchor_phrase"], p["category"]))
    from datasets import load_dataset

    if source == "tweet_eval":
        ds = load_dataset("cardiffnlp/tweet_eval", "offensive", split="test")
        texts = [r["text"] for r in ds if r["label"] == 1]
    else:  # hatecheck
        hc = load_hatecheck()
        texts = hc.loc[(hc["label_gold"] == "hateful") & ~hc["functionality"].str.startswith("spell_"),
                       "test_case"].tolist()
    texts = [" ".join(t.split()) for t in texts if 3 <= len(t.split()) <= 40]
    random.Random(SEED).shuffle(texts)
    return [(t, source) for t in texts[:n_candidates]]


def run_attack(recipe, classifier, source, n, n_candidates):
    import textattack
    from textattack.attack_recipes import BAEGarg2019, TextFoolerJin2019
    from textattack.attack_results import SkippedAttackResult, SuccessfulAttackResult
    from textattack.models.wrappers import ModelWrapper

    import nltk

    for res in ["averaged_perceptron_tagger_eng", "omw-1.4", "wordnet", "stopwords", "universal_tagset"]:
        nltk.download(res, quiet=True)  # TextAttack's POS filter needs these
    clf = ToxicityClassifier(classifier, THRESHOLD)

    class ThresholdWrapper(ModelWrapper):
        """Class 1 = alert. The attack succeeds when argmax != 1, i.e. exactly score < threshold."""

        def __init__(self):
            self.model = clf.model

        def __call__(self, texts):
            p = clf.score(list(texts))
            return np.stack([np.full_like(p, THRESHOLD) if THRESHOLD != 0.5 else 1 - p, p], axis=1)

    textattack.shared.utils.set_seed(SEED)
    attack = {"textfooler": TextFoolerJin2019, "bae": BAEGarg2019}[recipe].build(ThresholdWrapper())

    items = harmful_texts(source, n_candidates)
    scores = clf.score([t for t, _ in items])
    all_flagged = [(t, c) for (t, c), s in zip(items, scores) if s >= THRESHOLD]
    flagged = all_flagged[:n]
    print(f"{len(items)} harmful texts, {len(all_flagged)} flagged by {classifier}, attacking {len(flagged)} with {recipe}")

    rows, counts = [], {"success": 0, "failed": 0, "skipped": 0}
    for i, (text, category) in enumerate(flagged, 1):
        result = attack.attack(text, 1)
        if isinstance(result, SuccessfulAttackResult):
            counts["success"] += 1
            orig, adv = result.original_result.attacked_text, result.perturbed_result.attacked_text
            rows.append({"category": category, "anchor_phrase": text, "perturbed_phrase": adv.text,
                         "words_changed": len(orig.all_words_diff(adv)), "queries": result.perturbed_result.num_queries})
        elif isinstance(result, SkippedAttackResult):
            counts["skipped"] += 1
        else:
            counts["failed"] += 1
        if i % 10 == 0:
            print(f"  {i}/{len(flagged)} attacked, {counts['success']} successful")

    df = pd.DataFrame(rows)
    if len(df):
        df.insert(0, "pair_id", range(len(df)))
        df["source"] = f"{recipe}@{classifier}"
        df["gate_ok"] = [check_pair(a, p)[0] for a, p in zip(df["anchor_phrase"], df["perturbed_phrase"])]
    name = f"{source}_{recipe}_{classifier}"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    summary = {"recipe": recipe, "attacked_classifier": classifier, "source": source, "threshold": THRESHOLD,
               "harmful_texts": len(items), "flagged": len(all_flagged), "attacked": len(flagged), **counts,
               "attack_success_rate": counts["success"] / len(flagged) if flagged else None,
               "mean_words_changed": float(df["words_changed"].mean()) if len(df) else None,
               "mean_queries": float(df["queries"].mean()) if len(df) else None,
               "pairs_passing_gate": int(df["gate_ok"].sum()) if len(df) else 0}
    (OUT / f"{name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


parser = argparse.ArgumentParser()
parser.add_argument("step", choices=["hatecheck", "tweets", "attack"])
parser.add_argument("--per-functionality", type=int, default=10)
parser.add_argument("--n", type=int, default=300, help="tweets: how many; attack: max texts to attack")
parser.add_argument("--recipe", choices=["textfooler", "bae"])
parser.add_argument("--classifier", choices=list(CLASSIFIERS))
parser.add_argument("--source", choices=["curated", "tweet_eval", "hatecheck"], default="curated")
parser.add_argument("--candidates", type=int, default=3000, help="harmful texts scored before picking flagged ones")
args = parser.parse_args()

if args.step == "hatecheck":
    make_hatecheck(args.per_functionality)
elif args.step == "tweets":
    make_tweets(args.n)
else:
    if not args.recipe or not args.classifier:
        raise SystemExit("attack needs --recipe and --classifier")
    run_attack(args.recipe, args.classifier, args.source, args.n, args.candidates)
