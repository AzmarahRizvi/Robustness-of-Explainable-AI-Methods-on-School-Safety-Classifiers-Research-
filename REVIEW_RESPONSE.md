# Reviewer comments and what changed

"Code" = done in this repository. "Paper" = you still need to write it in the paper text.

## Reviewer yyTL

| Comment | What we did | Where |
|---|---|---|
| M1: score drops not tied to a detection decision | Code: one fixed label per model (`toxic`, `toxicity`), sigmoid, threshold 0.5. Report counts flagged anchors, evaded, still flagged, newly flagged, and repeats this at thresholds 0.1/0.3/0.5/0.7/0.9. Unflagged anchors are reported as a finding (with/without swear words). Paper: state label, sigmoid and threshold; say generic toxicity models detect abusive language, not intent. | `config.py`, `classifier.py`, report H1a |
| M2: pairs not validated | Code: automatic checks (embedding similarity, NLI both ways, % words changed, GPT-2 perplexity, word frequency, optional grammar errors) + 3 local LLM annotators (Qwen, Mistral, Gemma) on blinded, shuffled pairs with Fleiss' kappa and Krippendorff's alpha + optional human sheet. Paper: LLM ratings are a proxy for humans (cite BAE on automatic scores vs humans); release all 50 pairs in an appendix. | `scripts/05_quality_checks.py` |
| M3: XAI claim based on one example | Code: Pearson, Spearman, top-k overlap for every pair, both models, all 3 methods, mean and SD, split by alert flipped / unchanged, compared with a seed-to-seed noise floor. "Explanation failure" defined before results: main risk word in top 30% for the anchor but not after the swap. | `explainers.py`, `scripts/06_noise_floor.py`, report H1b |
| M4: SLM and XAI not on a common measure; judge used attributions | Code: one shared question for every method, "is the risk still identified after the swap?" (alert / risk word in top-k / SLM harmful + same category / narrative category from a judge that sees only the narrative). Judge now sees the raw query, never attributions. | report H1c, `scripts/04_judge_narratives.py` |
| M5: wrong test (unpaired Mann-Whitney) | Code: per anchor d = synonym change - control change; Wilcoxon signed-rank + sign-flip permutation; McNemar for evasion; Holm. v1 data re-analysed in `notebooks/results.ipynb`. | `stats_utils.py` |
| M6: not real school models; related work missing | Code: TextFooler and BAE through TextAttack, on our anchors and on tweet_eval / HateCheck. Paper: cite TextFooler, BAE, TextAttack, DNE, TEF, Ludan; say Gaggle / Linewize are not public, so we use public toxicity classifiers "of the type such systems may rely on". | `scripts/07_make_public_pairs.py` |
| "Trigram" had 4 words | Code: one word definition everywhere; windows have exactly 3 words (unit test). Paper: fix Table 1 and Figure 3 from `notebooks/case_study.ipynb`. | `explainers.ngram_windows` |
| Tokenizer explanation untested | Code: extra sub-word pieces per pair for each tokenizer, Spearman with score change. Paper: keep the claim only if significant. | report H1a |
| "200 judge calls" | Code: exact counts (narratives, judgements, failed calls) saved and printed. Each narrative is generated once and cached. | `judge_info.json` |
| Reproducibility details | Code: HF commit, library versions, seed, IG steps/baseline, SHAP/LIME budgets, spaCy model, GED formula, filler word list all saved or in `config.py`. Paper: release the repo anonymously (e.g. anonymous.4open.science). | `run_info.json`, `config.py` |
| No speed / memory numbers | Code: seconds per query (mean, median, p95) and Ollama memory for Llama-3-8B on this machine. | `slm_speed.json` |
| Privacy (gpt-4o-mini) | Code: OpenAI removed. Judge and annotators run locally through Ollama. Paper: say so. | `llm.py` |
| Narratives differ between BERT and RoBERTa | Code: the SLM is its own step (script 03), never sees the classifier, every answer is cached, so both classifiers share one set of narratives. | `scripts/03_run_slm.py` |

## Reviewer en7L

| Comment | What we did |
|---|---|
| SLM only sees flagged inputs | Runs on every query, independently of the classifier. Paper: fix the text and Figure 1. |
| Prompt leaks the label | The "flagged as high-risk" sentence is gone. The SLM writes the intent first, then decides risk (harmful/benign/unclear) and category (explain-then-predict, Ludan et al.). |
| No benign / ambiguous queries | 40 benign + ambiguous queries, plus HateCheck non-hateful tests and benign tweets; report gives false-alarm rates for classifiers and SLM. |
| SLM results not significant | New main SLM test is detection (McNemar on misses, recovery of classifier evasions); GED is now secondary. More data via HateCheck / TextFooler / BAE. |
| No data or code | Everything is in this repository. |

## AI reviewer

| Comment | What we did |
|---|---|
| Hypothesis not tested | Report is split into H1a, H1b, H1c, each with its own test. Paper: conclude each one separately. |
| Score drops are not false negatives | Same as yyTL M1. |
| Control confounded | All controls change the same number of words; `control_filler` and `control_wordnet` use the same positions; `control_nonrisk` changes other positions (TextFooler-style ablation). Regression of score change on condition + word-frequency change + extra sub-word pieces + perplexity increase, SEs clustered by anchor. |
| Wrong interpretation of IG | IG explains the sigmoid probability with a PAD baseline, 50 steps; completeness gap and convergence delta reported. Paper: remove "contradiction"; use "a single high IG segment can mislead a moderator who reads the chart without the overall score". |
| SLM can't act as a safety net | Same as en7L. |
| Novelty is narrow | Paper: "We do not propose a new attack. We study this known weakness in a safety-critical domain, measure its effect on the explanations moderators rely on, and test a privacy-preserving semantic analyser." Add a short task-definition section; call the SLM a semantic analyser throughout; use "toxicity score" consistently. |

## Known limitations to state in the paper

- LLM annotators are a proxy for human judgment.
- `control_wordnet` is valid for only 19/50 pairs and `control_nonrisk` for 28/50 (strict same-POS rule; spaCy tagging errors). Paired tests use only anchors where both versions exist.
- Many curated swaps remove a swear word; the profanity split and HateCheck show how much of the effect this explains.
- The earlier fully automatic WordNet / SLM generation failed (all senses, no meaning filter); this is why the 50 pairs were hand-curated, and why TextFooler/BAE (with their filters) are used for scale.

## Results to use in the paper

From `results/curated/report.md`, `results/textfooler_bert/report.md` and `notebooks/results.ipynb`.
Alert = sigmoid >= 0.5. "BERT" = Toxic-BERT, "RoBERTa" = Toxic-RoBERTa.

**H1a (supported).**
- Curated synonyms remove the alert in 31/37 flagged anchors for BERT (0.84, 95% CI 0.69-0.92) and 30/36 for RoBERTa (0.83).
  Meaning-keeping controls: non-risk positions 0.05, random WordNet synonym 0.29-0.47.
- Paired Wilcoxon (synonym vs control_nonrisk / control_wordnet / control_filler_random): all p < 0.003 after Holm.
  Same direction at every threshold 0.1-0.9.
- LLM-validated pairs only: 0.82 (BERT), 0.80 (RoBERTa).
- v1 data, paired re-analysis: median d = -0.148 (BERT), -0.115 (RoBERTa), Wilcoxon p = 2e-6 / 2e-5. The v1 conclusion holds with the correct test.
- Profanity: anchors with a swear word flagged 34/34 (BERT) and 32/34 (RoBERTa); without one 3/16 and 4/16 (Fisher p < 1e-6).
  Frame as "these models detect abusive language, not harmful intent".
- Tokenizer: rho = 0.11 / 0.10, p > 0.4. **Remove the tokenizer explanation.**
- Confound regression: synonym coefficient -0.25 for both models; p = 0.03 (BERT), p = 0.08 (RoBERTa).
- TextFooler (attacking BERT): 37/37 successful, about 1.1 words changed, about 51 queries; transfers to RoBERTa 23/34 (0.68).
  But it often changes the meaning ("some bitch" -> "some cuckold", "christians" -> "kristen").
  On the same 37 anchors, TextFooler texts are less natural (LLM naturalness 3.05 vs 3.50, p = 1e-4) and use much rarer words (Zipf change -2.30 vs -0.44, p = 3e-7).
  The LLM meaning score is lower (0.70 vs 0.82, p = 0.05), yet embedding similarity is HIGHER (0.82 vs 0.76) because fewer words change, i.e. automatic metrics miss the meaning drift (as BAE argued).

**H1b (supported).**
- Explanation failure (risk word leaves the top 30%) on synonym pairs: IG 0.42 / 0.54, SHAP 0.57 / 0.51, LIME 0.63 / 0.65 (BERT / RoBERTa). Controls: 0.04-0.26.
- Also when the alert does NOT change: 0.33-0.53.
- Synonym-pair Spearman far below the same-text noise floor for every method and model (Holm p < 0.001).
  IG and SHAP are deterministic here, so their floor is 1.0; LIME's floor is 0.92-0.97.
- IG completeness gap 0.007 / 0.003.

**H1c (supported).**
- SLM still says harmful after the swap: 43/47 (0.91). Classifier alert: 0.16-0.17. Risk word in the top 30%: 0.35-0.58.
- SLM catches 26/31 BERT evasions (0.84) and 25/30 RoBERTa evasions (0.83).
  Misses on synonym texts: classifier 43, SLM 5; McNemar p = 7e-12.
- False alarms: SLM 1/22 benign and 4/18 ambiguous (classifiers 0). The ambiguous ones include "what to do if my friend says they want to hurt themselves".
- Judge (Qwen, local): faithfulness mostly 3/5 ("core intent right but adds details"); mean 2.9 on harmful texts, 4.0 on benign.
- Cost: 43 s per query (median 38, p95 70), 5.05 GB RAM, CPU only.

**Pair validation (LLM panel, proxy for humans).**
- Curated: 34/50 pass (same meaning AND same severity). The 16 that fail are mostly swaps that remove a swear word: same intent, softer tone.
- Agreement is poor: meaning kappa -0.16, severity kappa -0.24, category kappa 0.50.
  Mistral is very strict, Gemma very lenient. Report this, and treat the panel as weak evidence; a human check is the fix.

## Still open (not code)

- Paper text: each "Paper:" item in the tables above.
- Figures: from `notebooks/case_study.ipynb` (3-word windows) and `notebooks/results.ipynb`.
- Recommended: 2 people rate 25 pairs (`05_quality_checks.py human-sheet`), because the LLM panel agreement is poor.
- Release the repository through an anonymous link.
- Optional: BAE, a second SLM, and the full HateCheck run (`results/hatecheck` is an early BERT-only run).
