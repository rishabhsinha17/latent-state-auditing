from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_'-]*")


class BagOfWordsNaiveBayes:
    """Small trainable text classifier for NLA explanations.

    This intentionally avoids heavyweight ML dependencies so it can run in the
    same lightweight eval scripts as the activation probes.
    """

    def __init__(self, alpha: float = 1.0, use_bigrams: bool = True) -> None:
        self.alpha = alpha
        self.use_bigrams = use_bigrams
        self.vocabulary: set[str] = set()
        self.class_token_counts: dict[bool, Counter[str]] = {True: Counter(), False: Counter()}
        self.class_totals: dict[bool, int] = {True: 0, False: 0}
        self.class_doc_counts: dict[bool, int] = {True: 0, False: 0}
        self.is_fit = False

    def fit(self, texts: list[str], labels: list[bool]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not any(labels) or all(labels):
            raise ValueError("need at least one positive and one negative example")
        self.vocabulary.clear()
        self.class_token_counts = {True: Counter(), False: Counter()}
        self.class_totals = {True: 0, False: 0}
        self.class_doc_counts = {True: 0, False: 0}
        for text, label in zip(texts, labels):
            features = Counter(_features(text, use_bigrams=self.use_bigrams))
            self.vocabulary.update(features)
            self.class_token_counts[label].update(features)
            self.class_totals[label] += sum(features.values())
            self.class_doc_counts[label] += 1
        self.is_fit = True

    def score(self, text: str) -> float:
        if not self.is_fit:
            raise RuntimeError("classifier must be fit before scoring")
        log_pos = self._log_prob(text, True)
        log_neg = self._log_prob(text, False)
        max_log = max(log_pos, log_neg)
        pos = math.exp(log_pos - max_log)
        neg = math.exp(log_neg - max_log)
        return pos / (pos + neg)

    def score_many(self, texts: list[str]) -> list[float]:
        return [self.score(text) for text in texts]

    def _log_prob(self, text: str, label: bool) -> float:
        total_docs = self.class_doc_counts[True] + self.class_doc_counts[False]
        class_prior = (self.class_doc_counts[label] + self.alpha) / (total_docs + 2 * self.alpha)
        log_prob = math.log(class_prior)
        vocab_size = max(1, len(self.vocabulary))
        denom = self.class_totals[label] + self.alpha * vocab_size
        for feature, count in Counter(_features(text, use_bigrams=self.use_bigrams)).items():
            token_count = self.class_token_counts[label][feature]
            log_prob += count * math.log((token_count + self.alpha) / denom)
        return log_prob


def normalize_text(text: str) -> str:
    return " ".join(_features(text, use_bigrams=False))


def _features(text: str, use_bigrams: bool) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    if not use_bigrams:
        return tokens
    bigrams = [f"{left}__{right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams
