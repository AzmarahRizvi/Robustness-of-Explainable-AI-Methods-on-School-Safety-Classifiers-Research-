"""Toxicity classifier with one fixed output label and a threshold.

Both models are multi-label, so every label has its own sigmoid. We always
use the sigmoid of the configured label (config.CLASSIFIERS[...]["label"]).
An alert fires when that score >= threshold. IG, SHAP and LIME explain this
same number.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import CLASSIFIERS, THRESHOLD


class ToxicityClassifier:
    def __init__(self, name, threshold=THRESHOLD):
        info = CLASSIFIERS[name]
        self.name = name
        self.label = info["label"]
        self.harm_labels = info["harm_labels"]
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(info["hf_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(info["hf_name"]).to(self.device).eval()

        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.label_index = self.labels.index(self.label)
        self.harm_index = [self.labels.index(lab) for lab in self.harm_labels]
        # Hugging Face commit of the downloaded weights, saved in run_info.json.
        self.commit = getattr(self.model.config, "_commit_hash", None)

    @torch.no_grad()
    def all_probs(self, texts, batch_size=32):
        """Sigmoid probability of every label. Shape [n_texts, n_labels]."""
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(list(texts[i:i + batch_size]), return_tensors="pt", padding=True,
                                   truncation=True, max_length=256).to(self.device)
            out.append(torch.sigmoid(self.model(**batch).logits).cpu().numpy())
        return np.concatenate(out)

    def score(self, texts):
        """The alert score: sigmoid of the fixed label, one number per text."""
        return self.all_probs(list(texts))[:, self.label_index]

    def score_one(self, text):
        probs = self.all_probs([text])[0]
        harm = probs[self.harm_index]
        return {
            "score": float(probs[self.label_index]),
            "flagged": bool(probs[self.label_index] >= self.threshold),
            "max_harm_score": float(harm.max()),
            "max_harm_label": self.harm_labels[int(harm.argmax())],
        }

    def n_subwords(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # ---- used by Integrated Gradients ----
    def position_ids(self, input_ids):
        """Give the PAD baseline the same positions as the input (RoBERTa counts positions from pad_id + 1)."""
        offset = 0
        if self.model.config.model_type == "roberta":
            offset = self.model.config.pad_token_id + 1
        return (torch.arange(input_ids.shape[1], device=input_ids.device) + offset).unsqueeze(0)

    def prob_from_ids(self, input_ids, attention_mask, position_ids):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).logits
        return torch.sigmoid(logits)[:, self.label_index]
