"""Statistics used in the paper.

The synonym and control versions come from the same anchor, so the data are
PAIRED. For each anchor we take d = synonym_change - control_change and test
whether d is centred on zero (Wilcoxon signed-rank and a sign-flip permutation
test). v1 used an unpaired Mann-Whitney test, which was wrong.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from .config import SEED


def wilson_ci(k, n, z=1.96):
    """95% confidence interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k, n):
    lo, hi = wilson_ci(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi}


def rank_biserial(d):
    """Effect size for the Wilcoxon test, between -1 and 1."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def bootstrap_ci(values, n_boot=10000, seed=SEED):
    """95% bootstrap interval for the median."""
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sign_flip_test(d, n_perm=20000, seed=SEED):
    """Permutation test for paired data: randomly flip the sign of each difference."""
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    flips = rng.choice([-1, 1], size=(n_perm, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def paired_test(x, y):
    """Paired comparison of x and y (same anchors). Drops pairs with a missing value."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    d = x[keep] - y[keep]
    out = {"n": int(len(d)), "mean_diff": float("nan"), "median_diff": float("nan"),
           "median_ci_low": float("nan"), "median_ci_high": float("nan"),
           "rank_biserial": float("nan"), "wilcoxon_p": float("nan"), "permutation_p": float("nan")}
    if len(d) == 0:
        return out
    lo, hi = bootstrap_ci(d)
    out.update(mean_diff=float(d.mean()), median_diff=float(np.median(d)), median_ci_low=lo, median_ci_high=hi,
               rank_biserial=rank_biserial(d))
    if np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
        out["permutation_p"] = sign_flip_test(d)
    return out


def mcnemar(a, b):
    """Exact McNemar test for two paired yes/no vectors."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first": only_a, "only_second": only_b, "both": int((a & b).sum()), "p": float(p)}


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    if keep.sum() < 3:
        return {"n": int(keep.sum()), "rho": float("nan"), "p": float("nan")}
    r = stats.spearmanr(x[keep], y[keep])
    return {"n": int(keep.sum()), "rho": float(r.statistic), "p": float(r.pvalue)}


def holm(pvalues):
    """Holm-Bonferroni correction. Input and output: {test name: p}."""
    items = sorted((p, name) for name, p in pvalues.items() if not np.isnan(p))
    out, running = {}, 0.0
    for i, (p, name) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[name] = running
    return out


# ---------------------------------------------------------------- annotator agreement
def fleiss_kappa(ratings):
    """ratings: 2D array [items, raters] of category labels (NaN = missing). Items with < 2 ratings are dropped."""
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa as fk

    rows = [r[~pd.isna(r).astype(bool)] for r in np.asarray(ratings, dtype=object)]
    n = min((len(r) for r in rows if len(r) >= 2), default=0)
    rows = [r[:n] for r in rows if len(r) >= n and n >= 2]
    if not rows:
        return float("nan")
    table, _ = aggregate_raters(np.array(rows, dtype=str))
    if table.shape[1] < 2:
        return float("nan")  # everyone gave the same answer: kappa undefined, see percent agreement
    return float(fk(table))


def krippendorff_alpha(ratings, level):
    """ratings: 2D numeric array [items, raters]. level: 'nominal' or 'ordinal'."""
    import krippendorff

    data = np.asarray(ratings, dtype=float).T  # krippendorff wants [raters, items]
    try:
        return float(krippendorff.alpha(reliability_data=data, level_of_measurement=level))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def percent_agreement(ratings):
    """Share of items where every rater gave the same answer."""
    same = []
    for r in np.asarray(ratings, dtype=object):
        r = r[~pd.isna(r).astype(bool)]
        if len(r) >= 2:
            same.append(len(set(map(str, r))) == 1)
    return float(np.mean(same)) if same else float("nan")

