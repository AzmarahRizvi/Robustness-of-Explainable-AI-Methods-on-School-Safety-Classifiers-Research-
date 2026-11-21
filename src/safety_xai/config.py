"""All settings in one place. Every number reported in the paper comes from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LLM_CACHE = RESULTS_DIR / "llm_cache.jsonl"

PAIRS_FILE = DATA_DIR / "student_phrase_pairs_50.csv"
BENIGN_FILE = DATA_DIR / "benign_queries.csv"

SEED = 42

# ---------------------------------------------------------------- classifiers
# "label" is the ONE sigmoid output we score, explain and threshold.
# "harm_labels" are only used for a secondary max-over-harm-labels score.
# Identity labels (e.g. "muslim" in Toxic-RoBERTa) are not harm labels.
CLASSIFIERS = {
    "toxic_bert": {
        "hf_name": "unitary/toxic-bert",
        "label": "toxic",
        "harm_labels": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "trained_on": ["jigsaw"],
    },
    "toxic_roberta": {
        "hf_name": "unitary/unbiased-toxic-roberta",
        "label": "toxicity",
        "harm_labels": ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat",
                        "sexual_explicit"],
        "trained_on": ["jigsaw", "civil_comments"],
    },
}

# Alert rule: sigmoid(label) >= THRESHOLD.
THRESHOLD = 0.5
THRESHOLDS_TO_TRY = [0.1, 0.3, 0.5, 0.7, 0.9]

# ---------------------------------------------------------------- conditions
# synonym                the curated (or attack-generated) synonym version
# control_filler         neutral nouns at the SAME positions as the synonym swap
#                        (this overwrites the risk word, so a drop is correct here)
# control_filler_random  neutral nouns at random positions (the v1 control)
# control_wordnet        random WordNet synonym at the SAME positions
# control_nonrisk        WordNet synonyms at OTHER (non-risk) content-word positions
#                        (TextFooler-style ablation: does any change move the score?)
CONDITIONS = ["synonym", "control_filler", "control_filler_random", "control_wordnet", "control_nonrisk"]

FILLER_WORDS = [
    "table", "window", "morning", "paper", "garden", "coffee", "bicycle", "mountain", "keyboard", "jacket",
    "kitchen", "printer", "calendar", "umbrella", "blanket", "pencil", "sidewalk", "hallway", "bookshelf",
    "sandwich",
]

RISK_CATEGORIES = [
    "threats_violence",
    "harassment_stalking",
    "hate_speech_gender",
    "hate_speech_religion",
    "hate_speech_ethnicity_nationality",
    "insults_degradation",
    "sexual_harassment",
    "cyberbullying_exclusion_rumors",
    "academic_sabotage_threats",
    "photo_video_shaming",
]

# ---------------------------------------------------------------- explanations
IG_STEPS = 50          # Integrated Gradients steps
SHAP_EVALS = 500       # SHAP partition explainer budget
LIME_SAMPLES = 500     # LIME perturbation samples
TOP_K = 0.3            # "risk word still in the top 30% of attributions"
NGRAM = 3              # window size for the figures (exactly 3 words)

# ---------------------------------------------------------------- local LLMs (Ollama)
SLM_MODEL = "llama3:8b"         # the semantic analyser under test
JUDGE_MODEL = "qwen2.5:7b"      # offline judge, different family from the SLM
ANNOTATOR_MODELS = ["qwen2.5:7b", "mistral:7b", "gemma2:9b"]  # none of them is Llama
LLM_OPTIONS = {"temperature": 0.0, "seed": SEED, "num_ctx": 2048, "num_predict": 200}

# ---------------------------------------------------------------- other models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
PERPLEXITY_MODEL = "gpt2"
SPACY_MODEL = "en_core_web_sm"


def run_dir(name):
    """results/<name>/ holds everything for one pair set (curated, hatecheck, ...)."""
    return RESULTS_DIR / name
