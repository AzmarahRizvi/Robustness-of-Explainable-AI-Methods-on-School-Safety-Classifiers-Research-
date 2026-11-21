# Synonym swaps, toxicity classifiers, XAI and a local SLM for school safety

Code and data for the paper *Bridging the Gap between Traditional Explainable AI and
SLM-Based Reasoning for School Safety Monitoring* (revised version).

We test three hypotheses:

- **H1a**: public toxicity classifiers (Toxic-BERT, Toxic-RoBERTa) lose their alert when a
  risky word is replaced by a synonym.
- **H1b**: the IG / SHAP / LIME explanations a moderator would read become unstable under the same swap.
- **H1c**: a small language model (Llama-3-8B, run locally) keeps identifying the risk better
  than the classifier.

## Folder layout

```
data/
  student_phrase_pairs_50.csv   50 hand-written anchor / synonym pairs, 10 risk categories
  benign_queries.csv            22 benign + 18 ambiguous queries (false alarms)
  generated/                    pairs made by script 07 (HateCheck, TextFooler, BAE)
src/safety_xai/                 shared code used by the scripts and notebooks
  config.py                     every setting (labels, threshold, models, seeds)
  data_utils.py                 loading, pair checks, the 4 control versions
  classifier.py                 classifier with one fixed label + threshold
  explainers.py                 IG, SHAP, LIME and the stability metrics
  llm.py                        local Ollama calls (cached) and the SLM prompt
  narratives.py                 graph and embedding similarity of SLM narratives
  stats_utils.py                paired tests, CIs, McNemar, Holm, Fleiss' kappa
scripts/                        run these in order (01 -> 08)
notebooks/
  case_study.ipynb              the one-pair figure for the paper
  results.ipynb                 v1 re-analysis + paper figures from the results
  report.ipynb                  every number and figure in the supervisor report (saved with outputs)
results/
  curated/                      everything for the 50 curated pairs (made by the scripts)
  v1_submission/                raw CSVs from the first submission (kept for the re-analysis)
  llm_cache.jsonl               every SLM / judge / annotator answer, so each is generated once
tests/test_basic.py             unit tests for the model-free code
REVIEW_RESPONSE.md              each reviewer comment and what changed
```

Everything in `results/` (except `v1_submission/`) and `data/generated/` is made by the
scripts and is not pushed to git. Run the scripts to make it again.

## Setup (once)

```powershell
uv sync                     # installs everything into .venv
uv run pytest               # 12 quick tests, should all pass
ollama pull llama3:8b       # the SLM under test
ollama pull qwen2.5:7b      # judge + annotator (different family from the SLM)
ollama pull mistral:7b      # annotator
ollama pull gemma2:9b       # annotator
```

Optional extras:

```powershell
uv sync --extra attacks     # TextFooler and BAE (TextAttack + TensorFlow, large download)
uv sync --extra grammar     # grammar-error count (also needs Java installed)
```

Ollama must be running (open the Ollama app) for scripts 03, 04 and 05 llm.

## How to get the results: run the Python scripts, then open the notebooks

**Run the scripts** for the experiment. They save their output after every pair, so
you can stop them (Ctrl+C) and start them again; finished work is skipped.
**Use the notebooks only at the end** to look at results and make the figures.
The notebooks do not run the experiment.

Run from the project folder, in this order. Times are for this laptop (8-core CPU, no GPU).

| Step | Command | What it makes | Time |
|---|---|---|---|
| 1 | `uv run python scripts/01_make_conditions.py` | `results/curated/conditions.csv`: the synonym and 4 control versions of each anchor | < 1 min |
| 2 | `uv run python scripts/02_run_classifier.py --classifier toxic_bert` | scores, alerts and IG/SHAP/LIME for every text | ~2 h |
| 2 | `uv run python scripts/02_run_classifier.py --classifier toxic_roberta` | same for RoBERTa | ~2 h |
| 3 | `uv run python scripts/03_run_slm.py` | SLM decision + intent sentence for every text, speed and memory | ~3 h |
| 4 | `uv run python scripts/04_judge_narratives.py` | local judge: faithfulness to the raw query + narrative category | ~4-5 h |
| 5 | `uv run python scripts/05_quality_checks.py auto` | automatic meaning / naturalness checks | ~5 min |
| 5 | `uv run python scripts/05_quality_checks.py llm` | 3 LLM annotators rate every pair (blinded) | ~2 h |
| 5 | `uv run python scripts/05_quality_checks.py agree` | Fleiss' kappa, Krippendorff's alpha, validated pairs | seconds |
| 6 | `uv run python scripts/06_noise_floor.py --classifier toxic_bert` (and `toxic_roberta`) | explanation self-agreement across seeds | ~1.5 h each |
| 8 | `uv run python scripts/08_analyze.py` | **`results/curated/report.md`** and `summary.json` | seconds |

Then open `notebooks/results.ipynb` (and `notebooks/case_study.ipynb` for the case-study figure)
and choose *Run All*. In VS Code pick the `.venv` Python as the kernel.

You can run script 08 at any time. It skips sections whose input does not exist yet.
Steps 2 and 3 can run at the same time in two terminals. Steps 4 and 6 are optional
but answer reviewer points (judge without OpenAI, noise floor).

Quick test of the whole pipeline before the long runs:

```powershell
uv run python scripts/02_run_classifier.py --classifier toxic_bert --limit 2
uv run python scripts/03_run_slm.py --limit 5
uv run python scripts/08_analyze.py
```

### Optional: two people rate a sample (strongly recommended)

```powershell
uv run python scripts/05_quality_checks.py human-sheet --raters rater1 rater2 --n 25
```

Fill in `results/curated/quality/ratings/human_rater1_synonym.csv` (and rater2), then run
`05_quality_checks.py agree` again. It reports agreement between the humans and between the
humans and the LLM panel.

### Scale-up with public data and standard attacks (step 7)

```powershell
# HateCheck: 436 human-validated spelling-variant pairs + labelled single tests
uv run python scripts/07_make_public_pairs.py hatecheck
uv run python scripts/01_make_conditions.py --pairs data/generated/hatecheck_spelling.csv --name hatecheck --queries data/generated/hatecheck_tests.csv --conditions synonym control_filler_random
uv run python scripts/02_run_classifier.py --classifier toxic_bert --name hatecheck --no-explain
uv run python scripts/02_run_classifier.py --classifier toxic_roberta --name hatecheck --no-explain
uv run python scripts/03_run_slm.py --name hatecheck --limit 300
uv run python scripts/08_analyze.py --name hatecheck

# TextFooler / BAE baselines on your own anchors (needs: uv sync --extra attacks)
uv run python scripts/07_make_public_pairs.py attack --recipe textfooler --classifier toxic_bert --source curated
uv run python scripts/01_make_conditions.py --pairs data/generated/curated_textfooler_toxic_bert.csv --name textfooler_bert --queries none
uv run python scripts/02_run_classifier.py --classifier toxic_bert --name textfooler_bert
uv run python scripts/02_run_classifier.py --classifier toxic_roberta --name textfooler_bert   # transfer to the other model
uv run python scripts/08_analyze.py --name textfooler_bert
```

For HateCheck, the `synonym` rows in the report are HateCheck's spelling variants
(character swap, deletion, leetspeak); its `queries` are labelled hateful / non-hateful tests.
Use `--recipe bae` the same way, and `--source tweet_eval --n 300` for a large public set.
The SLM is slow on a CPU (~37 s per query), so use `--limit` for big sets.

## Definitions used in the paper

| Item | Definition |
|---|---|
| Alert | sigmoid of one fixed label >= 0.5: `toxic` (Toxic-BERT), `toxicity` (Toxic-RoBERTa). Thresholds 0.1-0.9 are also reported. |
| Evasion | anchor raises an alert and the perturbed text does not |
| Conditions | `synonym` (curated); `control_filler` (neutral nouns at the same positions); `control_filler_random` (neutral nouns at random positions, the v1 control); `control_wordnet` (random same-POS WordNet synonym at the same positions); `control_nonrisk` (WordNet synonyms at other, non-risk positions) |
| Paired test | per anchor d = synonym change - control change; Wilcoxon signed-rank + sign-flip permutation test; rank-biserial effect size; McNemar for evasion; Holm correction |
| Explanation stability | Pearson, Spearman, top-10/30/50% overlap between anchor and perturbed attributions (TEF), split by whether the alert flipped |
| Explanation failure | the main swapped risk word is in the top 30% of attributions for the anchor but not for the synonym version |
| SLM detection | SLM says `harmful`; it runs on every query, never sees the classifier, and is prompted to explain first, then decide |
| Shared measure | "is the risk still identified after the swap?": alert still fires / risk word still in top 30% / SLM still `harmful` (same category) |
| Graph similarity | 1 - GED / (nodes + edges of both graphs), in [0, 1] (secondary measure) |
| IG | explains the sigmoid probability; PAD baseline; 50 steps; completeness gap and Captum convergence delta are reported |

All randomness uses seed 42. Model versions, library versions, hardware and timings are saved
in `results/<name>/<classifier>/run_info.json` and `results/<name>/slm_speed.json`.

## Privacy

Every model runs locally. The SLM, the judge and the annotators all run through Ollama
on this machine, so no student text leaves it. The judge and annotators are used only to
evaluate the research data; they are not part of the school-side pipeline.
