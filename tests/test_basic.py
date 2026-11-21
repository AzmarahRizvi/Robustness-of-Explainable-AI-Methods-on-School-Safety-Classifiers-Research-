"""Quick tests for the parts that do not need any model. Run: uv run pytest"""

import numpy as np
import spacy

from safety_xai import data_utils, explainers, narratives, stats_utils
from safety_xai.config import CONDITIONS, PAIRS_FILE, SPACY_MODEL


def test_check_pair():
    assert data_utils.check_pair("how to bully someone", "how to bully someone") == (False, "no_change")
    assert data_utils.check_pair("how to bully someone", "how to bully") == (False, "word_count_differs")
    assert data_utils.check_pair("how to bully someone", "How to bully someone")[0] is False
    assert data_utils.check_pair("how to bully someone", "how to intimidate someone") == (True, "ok")


def test_all_curated_pairs_pass_the_gate():
    pairs = data_utils.load_pairs(PAIRS_FILE)
    assert len(pairs) == 50
    for a, p in zip(pairs["anchor_phrase"], pairs["perturbed_phrase"]):
        assert data_utils.check_pair(a, p)[0], (a, p)


def test_controls_change_the_same_number_of_words():
    nlp = spacy.load(SPACY_MODEL)
    anchor = "how to threaten someone without getting caught"
    synonym = "how to menace someone without getting detected"
    rows = data_utils.make_conditions(3, anchor, synonym, nlp, CONDITIONS)
    assert [r["condition"] for r in rows] == CONDITIONS
    by_cond = {r["condition"]: r for r in rows}
    assert by_cond["control_filler"]["positions"] == [2, 6]
    assert data_utils.changed_positions(anchor, by_cond["control_filler"]["text"]) == [2, 6]
    for r in rows:
        if r["valid"]:
            assert len(data_utils.changed_positions(anchor, r["text"])) == 2
    nonrisk = by_cond["control_nonrisk"]
    if nonrisk["valid"]:
        assert not set(nonrisk["positions"]) & {2, 6}
    # the same seed gives the same controls every time
    again = data_utils.make_conditions(3, anchor, synonym, nlp, CONDITIONS)
    assert [r["text"] for r in rows] == [r["text"] for r in again]


def test_profanity():
    assert data_utils.has_profanity("how to fuck someone up")
    assert not data_utils.has_profanity("how to hurt someone badly")


def test_windows_have_exactly_n_words():
    text = "how to make someone feel worthless at school"
    wins = explainers.ngram_windows(text, np.arange(8, dtype=float), n=3)
    assert all(len(w.split()) == 3 for w, _ in wins)
    assert len(wins) == 6
    assert wins[0] == ("how to make", 3.0)


def test_token_to_word():
    text = "how to bully"
    spans = explainers.word_spans(text)
    offsets = [(0, 0), (0, 3), (4, 6), (7, 9), (9, 12), (0, 0)]  # [CLS] how to bu ##lly [SEP]
    assert explainers.token_to_word(offsets, spans) == [None, 0, 1, 2, 2, None]


def test_compare_explanations():
    a = np.array([0.1, 0.1, 0.9, 0.1, 0.05])
    same = explainers.compare(a, a.copy(), [2])
    assert abs(same["spearman"] - 1.0) < 1e-9 and same["top30_overlap"] == 1.0
    assert same["risk_in_topk_anchor"] and same["risk_in_topk_perturbed"]
    moved = explainers.compare(a, np.array([0.9, 0.5, 0.0, 0.1, 0.05]), [2])
    assert moved["risk_in_topk_anchor"] and not moved["risk_in_topk_perturbed"]


def test_paired_test_detects_a_shift():
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    x = y - 1.0 + rng.normal(scale=0.1, size=40)
    t = stats_utils.paired_test(x, y)
    assert t["wilcoxon_p"] < 0.001 and t["permutation_p"] < 0.001
    assert t["rank_biserial"] < -0.9


def test_mcnemar_and_holm():
    m = stats_utils.mcnemar([True] * 10 + [False] * 10, [False] * 20)
    assert m["only_first"] == 10 and m["p"] < 0.01
    adj = stats_utils.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert abs(adj["a"] - 0.03) < 1e-12 and abs(adj["b"] - 0.08) < 1e-12 and adj["c"] == 0.5


def test_agreement():
    perfect = np.array([["a", "a", "a"], ["b", "b", "b"], ["a", "a", "a"], ["c", "c", "c"]], dtype=object)
    assert stats_utils.percent_agreement(perfect) == 1.0
    assert abs(stats_utils.fleiss_kappa(perfect) - 1.0) < 1e-9
    numeric = np.array([[1, 1], [0, 0], [0.5, 0.5], [1, 1]], dtype=float)
    assert abs(stats_utils.krippendorff_alpha(numeric, "ordinal") - 1.0) < 1e-9


def test_graph_similarity_range():
    nlp = spacy.load(SPACY_MODEL)
    g1 = narratives.make_graph(nlp, "The student wants to hurt a classmate after school.")
    g2 = narratives.make_graph(nlp, "The student wants to find a library book.")
    same, _ = narratives.graph_similarity(g1, g1)
    diff, _ = narratives.graph_similarity(g1, g2)
    assert same == 1.0 and 0.0 <= diff < 1.0


def test_wilson_ci():
    lo, hi = stats_utils.wilson_ci(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert np.isnan(stats_utils.wilson_ci(0, 0)[0])
