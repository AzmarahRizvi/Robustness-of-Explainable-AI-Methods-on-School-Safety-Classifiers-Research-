"""Comparing two SLM narratives: sentence graphs (GED) and embedding similarity.

Graph similarity = 1 - GED / (nodes + edges of graph A + nodes + edges of graph B)

The denominator is the cost of deleting one graph and inserting the other, so
the score is always between 0 and 1. This is a secondary measure; the main SLM
result is whether it still detects the risk (script 08, H1c).
"""

import time

import networkx as nx
import numpy as np


def triplets(nlp, text):
    """(subject, relation, object) triplets from the dependency parse."""
    out = []
    for tok in nlp(text):
        if tok.pos_ == "VERB":
            subj = next((c.lemma_ for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")), "student")
            obj = next((c.lemma_ for c in tok.children if c.dep_ in ("dobj", "pobj", "attr", "xcomp")), "intent")
            out.append((subj.lower(), tok.lemma_.lower(), obj.lower()))
        elif tok.dep_ == "prep":
            obj = next((c.lemma_ for c in tok.children if c.dep_ == "pobj"), "context")
            out.append((tok.head.lemma_.lower(), tok.lemma_.lower(), obj.lower()))
        elif tok.dep_ == "amod":
            out.append((tok.head.lemma_.lower(), "is", tok.lemma_.lower()))
    return out


def make_graph(nlp, text):
    g = nx.DiGraph()
    for s, r, o in triplets(nlp, text):
        g.add_edge(s, o, relation=r)
    return g


def graph_similarity(g1, g2, timeout=30):
    total = g1.number_of_nodes() + g1.number_of_edges() + g2.number_of_nodes() + g2.number_of_edges()
    if total == 0:
        return 1.0, True
    start = time.perf_counter()
    ged = nx.graph_edit_distance(g1, g2, node_match=lambda a, b: True,
                                 edge_match=lambda a, b: a["relation"] == b["relation"], timeout=timeout)
    finished = time.perf_counter() - start < timeout  # False means GED is an upper bound
    if ged is None:
        ged = total
    return 1 - ged / total, finished


class Embedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def cosine(self, texts_a, texts_b):
        a = self.model.encode(list(texts_a), normalize_embeddings=True)
        b = self.model.encode(list(texts_b), normalize_embeddings=True)
        return np.sum(a * b, axis=1)
