"""Word-level IG, SHAP and LIME, and metrics that compare two explanations.

Every method returns ONE number per whitespace word, for the classifier's fixed
label. Because the anchor and the perturbed text have the same number of words,
the two explanations can be compared position by position.

Integrated Gradients setup (reported in the paper):
  output explained : sigmoid probability of the fixed label (not the logit)
  baseline         : [CLS]/<s> + [PAD] for every real token + [SEP]/</s>
  steps            : config.IG_STEPS (50)
  word score       : token attributions summed over the embedding, then over
                     the sub-word pieces of each word
  completeness     : sum of attributions should equal F(input) - F(baseline).
                     Captum's convergence delta is the gap; we save it.
"""

import math
import re

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from scipy import stats

from .config import IG_STEPS, LIME_SAMPLES, SEED, SHAP_EVALS, TOP_K
from .data_utils import words


def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def token_to_word(offsets, spans):
    """For each token (start, end) give the index of the word it belongs to, or None."""
    out = []
    for start, end in offsets:
        found = None
        if end > start:  # special tokens have empty offsets
            for i, (s, e) in enumerate(spans):
                if s <= start < e or start < s < end:
                    found = i
                    break
        out.append(found)
    return out


# ---------------------------------------------------------------- IG
def integrated_gradients(clf, text, steps=IG_STEPS):
    enc = clf.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=256)
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    enc = enc.to(clf.device)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    is_special = torch.tensor([[e <= s for s, e in offsets]], device=clf.device)
    baseline = torch.where(is_special, ids, torch.full_like(ids, clf.tokenizer.pad_token_id))
    positions = clf.position_ids(ids)

    lig = LayerIntegratedGradients(clf.prob_from_ids, clf.model.get_input_embeddings())
    attr, delta = lig.attribute(inputs=ids, baselines=baseline, additional_forward_args=(mask, positions),
                                n_steps=steps, return_convergence_delta=True)
    token_scores = attr.sum(dim=-1)[0].detach().cpu().numpy()

    spans = word_spans(text)
    word_scores = np.zeros(len(spans))
    for w, s in zip(token_to_word(offsets, spans), token_scores):
        if w is not None:
            word_scores[w] += s

    with torch.no_grad():
        f_input = clf.prob_from_ids(ids, mask, positions).item()
        f_base = clf.prob_from_ids(baseline, mask, positions).item()
    check = {"f_input": f_input, "f_baseline": f_base, "attr_sum": float(token_scores.sum()),
             "convergence_delta": float(delta.item())}
    return word_scores, check


# ---------------------------------------------------------------- SHAP
def shap_values(clf, text, max_evals=SHAP_EVALS, seed=SEED):
    import shap

    masker = shap.maskers.Text(r"\s+")  # one feature per whitespace word
    explainer = shap.Explainer(lambda xs: clf.score([str(x) for x in xs]), masker, seed=seed)
    result = explainer([text], max_evals=max_evals, silent=True)[0]
    feats = [str(f).strip() for f in result.data]
    vals = [v for f, v in zip(feats, result.values) if f]
    if [f for f in feats if f] != words(text):
        raise ValueError(f"SHAP words do not match the text: {text}")
    return np.array(vals, dtype=float)


# ---------------------------------------------------------------- LIME
def lime_weights(clf, text, samples=LIME_SAMPLES, seed=SEED):
    from lime.lime_text import LimeTextExplainer

    # bow=False: each POSITION is a feature, so a repeated word keeps separate weights.
    explainer = LimeTextExplainer(class_names=["no_alert", "alert"], bow=False, split_expression=r"\s+",
                                  random_state=seed)

    def predict(xs):
        p = clf.score(list(xs))
        return np.stack([1 - p, p], axis=1)

    n = len(words(text))
    exp = explainer.explain_instance(text, predict, num_features=n, num_samples=samples, labels=(1,))
    out = np.zeros(n)
    for i, w in exp.as_map()[1]:
        out[i] = w
    return out


METHODS = ["ig", "shap", "lime"]


def explain(clf, text, seed=SEED):
    ig, ig_check = integrated_gradients(clf, text)
    return {"ig": ig, "shap": shap_values(clf, text, seed=seed), "lime": lime_weights(clf, text, seed=seed),
            "ig_check": ig_check}


# ---------------------------------------------------------------- comparing explanations
def top_k_positions(scores, frac=TOP_K):
    """Positions of the top `frac` share of words, ranked by attribution TOWARDS the alert."""
    k = max(1, math.ceil(frac * len(scores)))
    return set(np.argsort(-np.asarray(scores))[:k].tolist())


def main_risk_word(anchor_scores, risk_positions):
    """Of the swapped words, the one the anchor explanation relies on most."""
    return max(risk_positions, key=lambda i: anchor_scores[i])


def top_k_overlap(a, b, frac):
    """Overlap of the most important positions by absolute value (as in TEF, Ivankay et al. 2022)."""
    k = max(1, math.ceil(frac * len(a)))
    top_a = set(np.argsort(-np.abs(a))[:k])
    top_b = set(np.argsort(-np.abs(b))[:k])
    return len(top_a & top_b) / k


def correlation(fn, a, b):
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(fn(a, b)[0])


def compare(anchor_scores, perturbed_scores, risk_positions):
    """All stability numbers for one pair and one method.

    risk_positions are the positions the SYNONYM version swapped (the risk words),
    for every condition. So for control_nonrisk the question is whether the
    explanation still points at the untouched risk word.
    """
    a = np.asarray(anchor_scores, dtype=float)
    b = np.asarray(perturbed_scores, dtype=float)
    out = {
        "pearson": correlation(stats.pearsonr, a, b),
        "spearman": correlation(stats.spearmanr, a, b),
        "top10_overlap": top_k_overlap(a, b, 0.1),
        "top30_overlap": top_k_overlap(a, b, 0.3),
        "top50_overlap": top_k_overlap(a, b, 0.5),
    }
    if risk_positions:
        main = main_risk_word(a, risk_positions)
        out["risk_in_topk_anchor"] = main in top_k_positions(a)
        out["risk_in_topk_perturbed"] = main in top_k_positions(b)
        out["topk_chance"] = len(top_k_positions(a)) / len(a)  # hit rate of a random ranking
    return out


def ngram_windows(text, scores, n=3):
    """Windows of exactly n words with their summed attribution (for figures)."""
    toks = words(text)
    if len(toks) <= n:
        return [(" ".join(toks), float(np.sum(scores)))]
    return [(" ".join(toks[i:i + n]), float(np.sum(scores[i:i + n]))) for i in range(len(toks) - n + 1)]
