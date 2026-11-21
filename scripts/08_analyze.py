"""Step 8: compute all statistics and write the report.

    uv run python scripts/08_analyze.py
    uv run python scripts/08_analyze.py --name hatecheck

Reads whatever earlier steps produced in results/<name>/ (missing parts are
skipped) and writes results/<name>/report.md and results/<name>/summary.json.

The report has one section per hypothesis, each with its own test:
  H1a  classifiers fail under synonym swaps
       -> threshold crossings, paired Wilcoxon / permutation test on
          (synonym change - control change), confound regression
  H1b  XAI explanations become unstable
       -> TEF-style correlations and top-k overlap, "explanation failure" rate,
          split by whether the alert flipped, compared with the noise floor
  H1c  the SLM keeps detecting the risk better than the classifier
       -> the same question for every method: "is the risk still identified
          after the synonym swap?", McNemar classifier vs SLM, false alarms
"""

import argparse
import json

import numpy as np
import pandas as pd

from safety_xai import stats_utils as su
from safety_xai.config import CLASSIFIERS, CONDITIONS, THRESHOLDS_TO_TRY, TOP_K, run_dir
from safety_xai.explainers import METHODS

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="curated")
args = parser.parse_args()
folder = run_dir(args.name)

report = []      # markdown lines
summary = {}     # numbers, saved as JSON
pvalues = {}     # primary tests, Holm-corrected at the end

# The expected answer for each group of texts
GOLD = {"synonym": "harmful", "control_nonrisk": "harmful (risk word kept)",
        "control_wordnet": "harmful if the random synonym fits", "control_filler": "unclear (risk word overwritten)",
        "control_filler_random": "unclear (may overwrite risk word)"}


def read(path):
    """Read a CSV, or None if an earlier step has not made it (or it is empty)."""
    if not path.exists() or path.stat().st_size < 5:
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "n/a"
        if v != 0 and abs(v) < 0.001:
            return f"{v:.2e}"
        return f"{v:.3f}"
    return str(v)


def table(df):
    """Markdown table from a DataFrame."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def ci(r):
    return f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"


def add(*lines):
    report.extend(lines)


def paired(df, col, cond_a, cond_b):
    """Values of `col` for two conditions, matched by anchor."""
    a = df[df["condition"] == cond_a].set_index("pair_id")[col]
    b = df[df["condition"] == cond_b].set_index("pair_id")[col]
    common = a.index.intersection(b.index)
    return a.loc[common].astype(float).to_numpy(), b.loc[common].astype(float).to_numpy()


# ---------------------------------------------------------------- load
conditions = pd.read_csv(folder / "conditions.csv")
queries = read(folder / "queries.csv")
classifiers = [c for c in CLASSIFIERS if (folder / c / "scores.csv").exists()]
scores = {c: pd.read_csv(folder / c / "scores.csv") for c in classifiers}
slm = read(folder / "slm.csv")
slm_pairs = read(folder / "slm_pairs.csv")
judge = read(folder / "judge.csv")
auto = read(folder / "quality" / "auto_metrics.csv")
validation = read(folder / "quality" / "pair_validation_llm_synonym.csv")
agreement = read(folder / "quality" / "agreement_synonym.csv")
conds = [c for c in CONDITIONS if c in set(conditions["condition"])]

add(f"# Results for `{args.name}`", "")

# ---------------------------------------------------------------- setup
add("## Setup", "")
add("- Alert rule: sigmoid probability of ONE fixed label >= threshold. Labels: "
    + ", ".join(f"{c} -> `{CLASSIFIERS[c]['label']}`" for c in CLASSIFIERS) + ".")
for c in classifiers:
    info_path = folder / c / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        add(f"- {c}: threshold {info['threshold']}, Hugging Face commit `{info['hf_commit']}`, "
            f"hardware: {info['hardware']['cpu']} ({info['hardware']['cores']} cores, {info['hardware']['ram_gb']} GB RAM, "
            f"GPU: {info['hardware']['gpu']}).")
        summary.setdefault("run_info", {})[c] = info
valid_counts = conditions.groupby("condition")["valid"].agg(["sum", "count"]).reindex(conds)
add("", "Valid rows per condition (invalid = automatic gate failed, e.g. no WordNet synonym):", "",
    table(valid_counts.reset_index().rename(columns={"sum": "valid", "count": "total"})))
summary["valid_rows"] = valid_counts["sum"].to_dict()

# ---------------------------------------------------------------- data quality
if auto is not None or agreement is not None:
    add("## Pair quality (automatic checks and LLM annotators)", "")
    if auto is not None:
        cols = [c for c in ["words_changed_pct", "embedding_cosine", "nli_both_ways", "perplexity_increase",
                            "grammar_errors_added", "word_freq_change"] if c in auto]
        q = auto.groupby("condition")[cols].mean().reindex(conds).reset_index()
        add("Means per condition:", "", table(q))
        summary["auto_metrics"] = q.to_dict(orient="records")
    if agreement is not None:
        add("Agreement between annotators (LLM panel = proxy for human judgment):", "", table(agreement))
        summary["agreement"] = agreement.to_dict(orient="records")
    if validation is not None:
        k, n = int(validation["passes"].sum()), len(validation)
        add(f"LLM panel: {k}/{n} synonym pairs judged 'similar' meaning AND 'same' severity by majority. "
            f"Mean meaning score {fmt(validation['meaning_score'].mean())} (1 = similar, 0.5 = ambiguous, 0 = dissimilar).", "")
        summary["validated_pairs"] = {"passes": k, "n": n}

# ================================================================= H1a
add("## H1a: classifiers fail under synonym swaps", "")
for c in classifiers:
    df = scores[c]
    thr = float(df["threshold"].iloc[0])
    s = summary.setdefault("H1a", {}).setdefault(c, {})
    add(f"### {c}", "")

    anchors = df.drop_duplicates("pair_id")
    n_flag = int(anchors["anchor_flagged"].sum())
    add(f"Anchors flagged at threshold {thr}: **{n_flag} / {len(anchors)}**. "
        f"The other {len(anchors) - n_flag} are never flagged, so they cannot be 'evaded'.", "")

    rows = []
    for cond in conds:
        d = df[df["condition"] == cond]
        flagged = d[d["anchor_flagged"]]
        r = su.rate(int(flagged["evaded"].sum()), len(flagged))
        rows.append({"condition": cond, "pairs": len(d), "anchors flagged": len(flagged),
                     "evaded (alert -> no alert)": r["k"], "evasion rate": r["rate"], "95% CI": ci(r),
                     "still flagged": int(flagged["text_flagged"].sum()),
                     "newly flagged (no alert -> alert)": int(d["newly_flagged"].sum()),
                     "mean score change": d["delta"].mean()})
        s[f"evasion_{cond}"] = r
    add("Threshold crossings:", "", table(pd.DataFrame(rows)))

    # different thresholds
    rows = []
    for t in THRESHOLDS_TO_TRY:
        row = {"threshold": t}
        for cond in conds:
            d = df[df["condition"] == cond]
            fl = d["anchor_score"] >= t
            ev = fl & (d["text_score"] < t)
            row[f"{cond}: flagged"] = int(fl.sum())
            row[f"{cond}: evasion rate"] = ev.sum() / fl.sum() if fl.sum() else float("nan")
        rows.append(row)
    sweep = pd.DataFrame(rows)
    keep = ["threshold"] + [col for col in sweep.columns if col.startswith("synonym")] + \
        [col for col in sweep.columns if "evasion" in col and not col.startswith("synonym")]
    add("Same table at other thresholds (does the result depend on 0.5?):", "", table(sweep[keep]))
    s["threshold_sweep"] = sweep.to_dict(orient="records")

    # paired test: synonym change - control change, per anchor
    rows = []
    for ctrl in [x for x in conds if x != "synonym"]:
        x, y = paired(df, "delta", "synonym", ctrl)
        t = su.paired_test(x, y)
        fs = df[(df["condition"] == "synonym") & df["anchor_flagged"]].set_index("pair_id")["evaded"]
        fc = df[(df["condition"] == ctrl) & df["anchor_flagged"]].set_index("pair_id")["evaded"]
        common = fs.index.intersection(fc.index)
        m = su.mcnemar(fs.loc[common], fc.loc[common])
        rows.append({"synonym vs": ctrl, "n": t["n"], "median of (syn change - ctrl change)": t["median_diff"],
                     "95% CI": f"[{fmt(t['median_ci_low'])}, {fmt(t['median_ci_high'])}]",
                     "rank-biserial": t["rank_biserial"], "Wilcoxon p": t["wilcoxon_p"],
                     "permutation p": t["permutation_p"],
                     "evaded only syn / only ctrl": f"{m['only_first']} / {m['only_second']}", "McNemar p": m["p"]})
        s[f"paired_vs_{ctrl}"] = {"score_change": t, "evasion_mcnemar": m}
        pvalues[f"H1a {c}: Wilcoxon syn vs {ctrl}"] = t["wilcoxon_p"]
    add("Paired test. For every anchor: d = (synonym score change) - (control score change). "
        "Negative d = the synonym lowers the score more than the control. Wilcoxon signed-rank and a sign-flip "
        "permutation test on the d values; McNemar on evasion among flagged anchors.", "", table(pd.DataFrame(rows)))
    add("Reading the controls: `control_filler` overwrites the risk word, so a big drop there is CORRECT behaviour. "
        "`control_wordnet` and `control_nonrisk` keep the meaning roughly intact; a big gap to them means the "
        "model reacts to the specific risk word, not to any change.", "")

    # profanity
    d = anchors.copy()
    rows = []
    for name, part in (("with profanity", d[d["anchor_profanity"]]), ("without profanity", d[~d["anchor_profanity"]])):
        r = su.rate(int(part["anchor_flagged"].sum()), len(part))
        rows.append({"anchors": name, "n": r["n"], "flagged": r["k"], "detection rate": r["rate"], "95% CI": ci(r)})
    from scipy.stats import fisher_exact

    tab = pd.crosstab(d["anchor_profanity"], d["anchor_flagged"]).reindex(index=[True, False], columns=[True, False],
                                                                        fill_value=0)
    fisher_p = float(fisher_exact(tab.to_numpy())[1])
    add("Detection of anchors with and without swear words. Generic toxicity models detect abusive LANGUAGE, "
        "not harmful INTENT:", "", table(pd.DataFrame(rows)), f"Fisher exact p = {fmt(fisher_p)}", "")
    s["profanity_detection"] = {"table": rows, "fisher_p": fisher_p}

    syn = df[(df["condition"] == "synonym") & df["anchor_flagged"]]
    removed = syn["anchor_profanity"] & ~syn["text_profanity"]
    rows = []
    for name, part in (("swap removed a swear word", syn[removed]), ("no swear word removed", syn[~removed])):
        r = su.rate(int(part["evaded"].sum()), len(part))
        rows.append({"flagged synonym pairs": name, "n": r["n"], "evaded": r["k"], "evasion rate": r["rate"],
                     "95% CI": ci(r)})
    add("Evasion split by whether the synonym swap removed a swear word:", "", table(pd.DataFrame(rows)))
    s["profanity_evasion"] = rows

    # tokenizer fragmentation
    syn_all = df[df["condition"] == "synonym"]
    t = su.spearman(syn_all["extra_subwords"], syn_all["delta"])
    add(f"Tokenizer check: Spearman between extra sub-word pieces (synonym - anchor) and score change: "
        f"rho = {fmt(t['rho'])}, p = {fmt(t['p'])}, n = {t['n']}. "
        "If this is not significant, drop the tokenizer explanation from the paper.", "")
    s["tokenizer_spearman"] = t

    # confound regression
    if auto is not None:
        import statsmodels.formula.api as smf

        reg = df.merge(auto[["pair_id", "condition", "word_freq_change", "perplexity_increase"]],
                       on=["pair_id", "condition"], how="inner")
        reg = reg[reg["condition"].isin(["synonym", "control_wordnet", "control_nonrisk"])].dropna(
            subset=["delta", "word_freq_change", "perplexity_increase", "extra_subwords"])
        if reg["condition"].nunique() >= 2 and "control_wordnet" in set(reg["condition"]):
            model = smf.ols("delta ~ C(condition, Treatment('control_wordnet')) + word_freq_change + extra_subwords"
                            " + perplexity_increase", data=reg).fit(cov_type="cluster",
                                                                    cov_kwds={"groups": reg["pair_id"]})
            coef = pd.DataFrame({"term": model.params.index.str.replace("C(condition, Treatment('control_wordnet'))",
                                                                         "condition", regex=False),
                                 "coefficient": model.params.values, "std error": model.bse.values,
                                 "p": model.pvalues.values})
            add("Confound check. OLS of score change on condition (reference: random WordNet synonym at the same "
                "positions) plus word-frequency change, extra sub-word pieces and GPT-2 perplexity increase; "
                f"standard errors clustered by anchor; n = {int(model.nobs)}. If the synonym coefficient stays "
                "negative and significant, the effect is not explained by these confounds.", "", table(coef))
            s["confound_regression"] = coef.to_dict(orient="records")

    # sensitivity: only LLM-validated pairs
    if validation is not None:
        ok_ids = set(validation.loc[validation["passes"], "pair_id"])
        v = syn[syn["pair_id"].isin(ok_ids)]
        r = su.rate(int(v["evaded"].sum()), len(v))
        add(f"Only synonym pairs the LLM panel validated: {r['k']}/{r['n']} flagged anchors evaded "
            f"(rate {fmt(r['rate'])}, CI {ci(r)}).", "")
        s["evasion_validated_only"] = r

# ================================================================= H1b
add("## H1b: XAI explanations become unstable", "")
add(f"Explanation failure (defined before looking at results): the swapped risk word is in the top {int(TOP_K * 100)}% "
    "of attributions (ranked towards the alert) for the anchor, but NOT for the synonym version. Stability is "
    "split by whether the alert flipped: when it flips, a smaller attribution may simply be a faithful "
    "explanation of a smaller score; true fragility is change while the decision stays the same.", "")
for c in classifiers:
    df = scores[c]
    if "ig_spearman" not in df:
        add(f"{c}: run without explanations, skipped.", "")
        continue
    s = summary.setdefault("H1b", {}).setdefault(c, {})
    add(f"### {c}", "")
    rows = []
    for m in METHODS:
        for cond in conds:
            d = df[df["condition"] == cond]
            for subset, part in (("all", d), ("alert unchanged", d[~d["alert_flipped"]]),
                                 ("alert flipped", d[d["alert_flipped"]])):
                if len(part) == 0:
                    continue
                in_a = part[f"{m}_risk_in_topk_anchor"].astype(bool)
                in_p = part[f"{m}_risk_in_topk_perturbed"].astype(bool)
                fail = su.rate(int((in_a & ~in_p).sum()), int(in_a.sum()))
                rows.append({"method": m, "condition": cond, "subset": subset, "n": len(part),
                             "Pearson mean": part[f"{m}_pearson"].mean(), "Pearson sd": part[f"{m}_pearson"].std(),
                             "Spearman mean": part[f"{m}_spearman"].mean(), "Spearman sd": part[f"{m}_spearman"].std(),
                             "top-30% overlap": part[f"{m}_top30_overlap"].mean(),
                             "risk word in top-k (anchor)": in_a.mean(),
                             "risk word in top-k (perturbed)": in_p.mean(),
                             "chance level": part[f"{m}_topk_chance"].mean(),
                             "explanation failures": f"{fail['k']}/{fail['n']}", "failure rate": fail["rate"]})
    stab = pd.DataFrame(rows)
    s["stability"] = stab.to_dict(orient="records")
    add(table(stab[stab["condition"].isin(["synonym", "control_wordnet", "control_nonrisk"])]))
    add("(Full table for every condition is in summary.json.)", "")

    noise = read(folder / c / "noise_floor.csv")
    if noise is not None:
        syn = df[df["condition"] == "synonym"].set_index("pair_id")
        rows = []
        for m in METHODS:
            nf = noise[noise["method"] == m].set_index("pair_id")["spearman"]
            for subset, part in (("all", syn), ("alert unchanged", syn[~syn["alert_flipped"]])):
                common = part.index.intersection(nf.index)
                x, y = part.loc[common, f"{m}_spearman"].to_numpy(float), nf.loc[common].to_numpy(float)
                t = su.paired_test(x, y)
                rows.append({"method": m, "subset": subset, "n": t["n"], "same-text Spearman (noise floor)": np.nanmean(y),
                             "synonym-pair Spearman": np.nanmean(x), "share of pairs below floor": np.nanmean(x < y),
                             "Wilcoxon p": t["wilcoxon_p"]})
                if subset == "alert unchanged":
                    pvalues[f"H1b {c}: {m} synonym vs noise floor"] = t["wilcoxon_p"]
        add("Compared with the noise floor (the same anchor explained with different seeds). Instability only "
            "counts where synonym-pair agreement is clearly below this floor:", "", table(pd.DataFrame(rows)))
        s["noise_floor"] = rows

    gap = (df["text_ig_attr_sum"] - df["text_ig_f_diff"]).abs()
    add(f"IG completeness check: mean |sum of attributions - (F(input) - F(baseline))| = {fmt(gap.mean())}, "
        f"mean |Captum convergence delta| = {fmt(df['text_ig_delta'].abs().mean())}. IG explains the sigmoid "
        "probability with a PAD baseline. Positive and negative attributions add up to the score difference, so "
        "one large segment next to negative ones is not a contradiction; it can mislead a moderator who reads the "
        "chart without the overall score.", "")
    s["ig_completeness_gap"] = float(gap.mean())

# ================================================================= H1c
add("## H1c: does the SLM keep detecting the risk better than the classifier?", "")
if slm_pairs is None:
    add("SLM outputs not found (run script 03).", "")
else:
    s = summary.setdefault("H1c", {})
    sp = slm_pairs.copy()
    sp["anchor_harmful"] = sp["anchor_risk"] == "harmful"
    sp["text_harmful"] = sp["text_risk"] == "harmful"
    if judge is not None:
        cat = judge.set_index("text")["narrative_category"]
        sp["anchor_narr_cat"] = sp["anchor"].map(cat)
        sp["text_narr_cat"] = sp["text"].map(cat)

    # shared question: "is the risk still identified after the synonym swap?"
    syn_slm = sp[sp["condition"] == "synonym"]
    rows = []

    def retention(name, identified_anchor, identified_text):
        a = np.asarray(identified_anchor, bool)
        t = np.asarray(identified_text, bool)
        r = su.rate(int((a & t).sum()), int(a.sum()))
        rows.append({"method": name, "pairs": len(a), "risk identified on anchor": int(a.sum()),
                     "still identified after swap": r["k"], "retention": r["rate"], "95% CI": ci(r),
                     "identified on synonym text (all pairs)": t.mean()})

    for c in classifiers:
        d = scores[c][scores[c]["condition"] == "synonym"]
        retention(f"{c}: alert", d["anchor_flagged"], d["text_flagged"])
        if "ig_risk_in_topk_anchor" in d:
            for m in METHODS:
                retention(f"{c}: {m.upper()} risk word in top {int(TOP_K * 100)}%",
                          d[f"{m}_risk_in_topk_anchor"].astype(bool), d[f"{m}_risk_in_topk_perturbed"].astype(bool))
    retention("SLM: harmful", syn_slm["anchor_harmful"], syn_slm["text_harmful"])
    retention("SLM: harmful + same category", syn_slm["anchor_harmful"],
              syn_slm["text_harmful"] & (syn_slm["anchor_slm_category"] == syn_slm["text_slm_category"]))
    if judge is not None:
        named = syn_slm["anchor_narr_cat"].fillna("none") != "none"
        retention("SLM narrative names same category (judge, narrative only)", named,
                  named & (syn_slm["anchor_narr_cat"] == syn_slm["text_narr_cat"]))
    add("One question for every method: **does it still point to the risk after the synonym swap?** "
        "Classifier = alert still fires; XAI = swapped risk word still in the top-k attributions; "
        "SLM = still labelled harmful (and same category).", "", table(pd.DataFrame(rows)))
    s["retention"] = rows

    # classifier vs SLM on the same synonym texts (all are harmful by construction)
    rows = []
    for c in classifiers:
        d = scores[c][scores[c]["condition"] == "synonym"][["pair_id", "text_flagged", "evaded"]]
        m = d.merge(syn_slm[["pair_id", "text_harmful"]], on="pair_id")
        mc = su.mcnemar(~m["text_flagged"].astype(bool), ~m["text_harmful"].astype(bool))
        ev = m[m["evaded"]]
        rec = su.rate(int(ev["text_harmful"].sum()), len(ev))
        rows.append({"classifier": c, "pairs": len(m), "classifier misses": int((~m["text_flagged"]).sum()),
                     "SLM misses": int((~m["text_harmful"]).sum()),
                     "missed only by classifier / only by SLM": f"{mc['only_first']} / {mc['only_second']}",
                     "McNemar p": mc["p"], "classifier evasions": rec["n"], "caught by SLM": rec["k"],
                     "recovery rate": rec["rate"], "95% CI": ci(rec)})
        pvalues[f"H1c {c}: McNemar misses classifier vs SLM"] = mc["p"]
    add("Misses on synonym texts (all harmful), classifier vs SLM, and how many classifier evasions the SLM "
        "still catches:", "", table(pd.DataFrame(rows)))
    s["classifier_vs_slm"] = rows

    # false negatives / false positives for every group of texts
    by_text = slm.set_index("text") if slm is not None else None
    rows = []

    def group_row(group, gold, texts, flags):
        texts = list(texts)
        risks = by_text.reindex(texts)["risk"] if by_text is not None else pd.Series([None] * len(texts))
        row = {"texts": group, "gold": gold, "n": len(texts)}
        for c, f in flags.items():
            row[f"{c} alert rate"] = float(np.mean(f)) if len(f) else float("nan")
        row["SLM harmful rate"] = float((risks == "harmful").mean())
        row["SLM unclear rate"] = float((risks == "unclear").mean())
        rows.append(row)

    first = scores[classifiers[0]] if classifiers else None
    if first is not None:
        anchors = first.drop_duplicates("pair_id")
        group_row("anchors", "harmful", anchors["anchor"],
                  {c: scores[c].drop_duplicates("pair_id")["anchor_flagged"] for c in classifiers})
        for cond in conds:
            gold = GOLD[cond]
            group_row(cond, gold, first.loc[first["condition"] == cond, "text"],
                      {c: scores[c].loc[scores[c]["condition"] == cond, "text_flagged"] for c in classifiers})
    if queries is not None:
        for qt in queries["query_type"].unique():
            flags = {}
            for c in classifiers:
                qs = read(folder / c / "queries_scores.csv")
                if qs is not None:
                    flags[c] = qs.loc[qs["query_type"] == qt, "flagged"]
            group_row(f"single queries: {qt}", qt, queries.loc[queries["query_type"] == qt, "query"], flags)
    fnfp = pd.DataFrame(rows)
    add("Alert rates for every group of texts. On harmful texts, 1 - rate = false-negative rate; "
        "on benign texts, the rate = false-positive rate. The SLM counts as an alert when it says 'harmful'.", "",
        table(fnfp))
    s["alert_rates"] = fnfp.to_dict(orient="records")

    # narrative stability (secondary)
    if "graph_similarity" in sp:
        rows = []
        for col in ("graph_similarity", "embedding_similarity"):
            for ctrl in [x for x in conds if x != "synonym"]:
                x, y = paired(sp.dropna(subset=[col]), col, "synonym", ctrl)
                t = su.paired_test(x, y)
                rows.append({"measure": col, "synonym mean": np.mean(x) if len(x) else float("nan"),
                             "vs": ctrl, "control mean": np.mean(y) if len(y) else float("nan"), "n": t["n"],
                             "median diff": t["median_diff"], "Wilcoxon p": t["wilcoxon_p"]})
        add("Secondary: how similar are anchor and perturbed narratives? (graph similarity = 1 - GED / total size)",
            "", table(pd.DataFrame(rows)))
        s["narrative_similarity"] = rows

    # judge
    if judge is not None:
        jt = judge.set_index("text")
        rows = []
        groups = [("anchors", sp["anchor"].drop_duplicates())]
        groups += [(cond, sp.loc[sp["condition"] == cond, "text"]) for cond in conds]
        if queries is not None:
            groups += [(f"single queries: {qt}", queries.loc[queries["query_type"] == qt, "query"])
                       for qt in queries["query_type"].unique()]
        for name, texts in groups:
            j = jt.reindex(list(texts))
            rows.append({"texts": name, "narratives judged": int(j["faithfulness"].notna().sum()),
                         "faithfulness (1-5)": j["faithfulness"].mean(),
                         "overstates risk": j["overstates_risk"].dropna().astype(bool).mean()})
        info_path = folder / "judge_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        add(f"Offline judge ({info.get('judge_model', '?')}, local; sees the raw query, never attribution scores). "
            f"{info.get('narratives', '?')} unique narratives, {info.get('total_judgements', '?')} judgements, "
            f"{info.get('failed_calls', '?')} failed calls (failed calls are excluded).", "", table(pd.DataFrame(rows)))
        s["judge"] = {"table": rows, "info": info}

    speed_path = folder / "slm_speed.json"
    if speed_path.exists():
        sp_info = json.loads(speed_path.read_text())
        t = sp_info.get("seconds_per_query_all_answers") or sp_info.get("seconds_per_query_new_calls")
        add(f"SLM cost on this machine ({sp_info['machine']['cpu']}, {sp_info['machine']['ram_gb']} GB RAM): "
            f"mean {fmt(t['mean'])} s per query (median {fmt(t['median'])}, p95 {fmt(t['p95'])}, n = {t['n']}). "
            f"Ollama memory: {sp_info['ollama_loaded_models']}.", "")
        s["slm_speed"] = sp_info

# ---------------------------------------------------------------- multiple testing
if pvalues:
    adj = su.holm(pvalues)
    add("## Holm-corrected p-values for the primary tests", "",
        table(pd.DataFrame({"test": list(adj), "raw p": [pvalues[k] for k in adj], "Holm p": list(adj.values())})))
    summary["holm"] = adj

(folder / "report.md").write_text("\n".join(report), encoding="utf-8")
(folder / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item")
                                                else str(o)), encoding="utf-8")
print(f"Wrote {folder / 'report.md'} and summary.json")
