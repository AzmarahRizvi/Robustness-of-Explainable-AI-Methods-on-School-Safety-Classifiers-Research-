"""Loading phrase pairs, checking them, and building the control versions."""

import hashlib
import random
import re

import pandas as pd

from .config import FILLER_WORDS, SEED

PUNCT_AT_END = re.compile(r"[,.!?\"';:]+$")


def words(text):
    """A word is anything between spaces. Every part of the code uses this definition."""
    return text.split()


def stable_seed(*parts):
    """Same seed on every machine and every run (Python's hash() is random per run)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- loading
def load_pairs(path):
    df = pd.read_csv(path)
    for col in ["category", "anchor_phrase", "perturbed_phrase"]:
        if col not in df.columns:
            raise ValueError(f"{path} needs a '{col}' column")
        df[col] = df[col].astype(str).str.strip()
    if "pair_id" not in df.columns:
        df.insert(0, "pair_id", range(len(df)))
    if "source" not in df.columns:
        df["source"] = "human_curated"
    return df


def load_queries(path):
    """Extra single queries (benign, ambiguous or labelled public tests). Needs 'query' and 'query_type'."""
    df = pd.read_csv(path)
    if "query" not in df.columns or "query_type" not in df.columns:
        raise ValueError(f"{path} needs 'query' and 'query_type' columns")
    if "topic" not in df.columns:
        df["topic"] = df["query_type"]
    return df


# ---------------------------------------------------------------- checks
def changed_positions(anchor, perturbed):
    """Indexes of the words that differ, or None if the word counts differ."""
    a, p = words(anchor), words(perturbed)
    if len(a) != len(p):
        return None
    return [i for i in range(len(a)) if a[i].lower() != p[i].lower()]


def check_pair(anchor, perturbed):
    """Catches mechanical mistakes only. Meaning is checked in script 05."""
    if " ".join(anchor.lower().split()) == " ".join(perturbed.lower().split()):
        return False, "no_change"
    positions = changed_positions(anchor, perturbed)
    if positions is None:
        return False, "word_count_differs"
    if not positions:
        return False, "case_only_change"
    return True, "ok"


# ---------------------------------------------------------------- profanity
# Jigsaw-trained models react strongly to swear words, and most curated swaps
# remove one. We split results by this list so that effect is visible.
PROFANITY_STEMS = ["fuck", "shit", "bitch", "cunt", "whore", "slut", "retard", "faggot", "asshole", "dumbass",
                   "jackass", "bastard", "wank", "twat", "dickhead", "pussy", "pussies"]
PROFANITY_WORDS = {"ass", "arse", "damn", "damned", "crap", "dick", "dicks", "piss", "pissed", "prick", "cock",
                   "fag", "fags", "hoe", "hoes", "tits", "bollocks", "douche", "douchebag", "hell"}


def has_profanity(text):
    for w in words(text):
        core = re.sub(r"[^a-z]", "", w.lower())
        if core in PROFANITY_WORDS or any(stem in core for stem in PROFANITY_STEMS):
            return True
    return False


# ---------------------------------------------------------------- WordNet
def get_wordnet():
    import nltk

    try:
        from nltk.corpus import wordnet
        wordnet.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet
    return wordnet


def get_stopwords():
    import nltk

    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


SPACY_TO_WORDNET = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def wordnet_synonyms(word, spacy_pos):
    """Single-word synonyms from the 2 most common senses only.

    Using all senses gives word-sense errors (e.g. "frighten" -> "dash"),
    which is why the earlier WordNet attempt failed.
    """
    wn = get_wordnet()
    pos = SPACY_TO_WORDNET.get(spacy_pos)
    if pos is None:
        return []
    out = set()
    for synset in wn.synsets(word.lower(), pos=pos)[:2]:
        for name in synset.lemma_names():
            if name.lower() != word.lower() and name.isalpha():
                out.add(name)
    return sorted(out)


def word_pos_tags(text, nlp):
    """spaCy part-of-speech tag for each whitespace word."""
    tags = [None] * len(words(text))
    starts = []
    pos = 0
    for w in words(text):
        pos = text.index(w, pos)
        starts.append((pos, pos + len(w)))
        pos += len(w)
    for token in nlp(text):
        for i, (s, e) in enumerate(starts):
            if s <= token.idx < e and tags[i] is None and not token.is_punct:
                tags[i] = token.pos_
    return tags


# ---------------------------------------------------------------- controls
def keep_case_and_punct(original, new_word):
    core = PUNCT_AT_END.sub("", original)
    end = original[len(core):]
    if core[:1].isupper():
        new_word = new_word[:1].upper() + new_word[1:]
    return new_word + end


def put_fillers(anchor, positions, seed):
    rng = random.Random(seed)
    toks = words(anchor)
    for i in positions:
        toks[i] = keep_case_and_punct(toks[i], rng.choice(FILLER_WORDS))
    return " ".join(toks)


def put_wordnet_synonyms(anchor, positions, tags, seed):
    """Returns the new text and how many positions really got a synonym."""
    rng = random.Random(seed)
    toks = words(anchor)
    n_done = 0
    for i in positions:
        options = wordnet_synonyms(PUNCT_AT_END.sub("", toks[i]), tags[i])
        if options:
            toks[i] = keep_case_and_punct(toks[i], rng.choice(options))
            n_done += 1
    return " ".join(toks), n_done


def nonrisk_positions(anchor, risk_positions, tags, n, seed):
    """n content words that are NOT the swapped risk words and have a WordNet synonym."""
    stop = get_stopwords()
    toks = words(anchor)
    options = []
    for i, w in enumerate(toks):
        core = PUNCT_AT_END.sub("", w)
        if i in risk_positions or core.lower() in stop or not core.isalpha() or has_profanity(core):
            continue
        if wordnet_synonyms(core, tags[i]):
            options.append(i)
    if len(options) < n:
        return None
    return sorted(random.Random(seed).sample(options, n))


def make_conditions(pair_id, anchor, synonym, nlp, conditions):
    """Build every perturbed version of one anchor.

    Returns a list of dicts: condition, text, positions, valid, reason.
    All controls change the same NUMBER of words as the synonym version.
    """
    ok, reason = check_pair(anchor, synonym)
    rows = []
    if not ok:
        for cond in conditions:
            rows.append({"condition": cond, "text": anchor, "positions": [], "valid": False,
                         "reason": "pair_" + reason})
        return rows

    pos = changed_positions(anchor, synonym)
    n = len(pos)
    tags = word_pos_tags(anchor, nlp)

    for cond in conditions:
        row = {"condition": cond, "text": anchor, "positions": pos, "valid": True, "reason": "ok"}
        if cond == "synonym":
            row["text"] = synonym
        elif cond == "control_filler":
            row["text"] = put_fillers(anchor, pos, stable_seed(SEED, pair_id, "filler"))
        elif cond == "control_filler_random":
            rng = random.Random(stable_seed(SEED, pair_id, "random_positions"))
            row["positions"] = sorted(rng.sample(range(len(words(anchor))), n))
            row["text"] = put_fillers(anchor, row["positions"], stable_seed(SEED, pair_id, "filler_random"))
        elif cond == "control_wordnet":
            row["text"], n_done = put_wordnet_synonyms(anchor, pos, tags, stable_seed(SEED, pair_id, "wordnet"))
            if n_done < n:
                row.update(valid=False, reason=f"no_wordnet_synonym_for_{n - n_done}_of_{n}")
        elif cond == "control_nonrisk":
            other = nonrisk_positions(anchor, pos, tags, n, stable_seed(SEED, pair_id, "nonrisk_positions"))
            if other is None:
                row.update(valid=False, reason="not_enough_nonrisk_words")
            else:
                row["positions"] = other
                row["text"], _ = put_wordnet_synonyms(anchor, other, tags, stable_seed(SEED, pair_id, "nonrisk"))
        else:
            raise ValueError(f"unknown condition {cond}")
        rows.append(row)
    return rows

