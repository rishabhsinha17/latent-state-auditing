from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable


@dataclass
class BinaryMetrics:
    auroc: float
    auprc: float
    fpr_at_90_tpr: float
    accuracy_at_50: float


@dataclass
class BootstrapInterval:
    mean: float
    lower: float
    upper: float
    samples: int


def binary_metrics(scores: list[float], labels: list[bool]) -> BinaryMetrics:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    return BinaryMetrics(
        auroc=auroc(scores, labels),
        auprc=auprc(scores, labels),
        fpr_at_90_tpr=fpr_at_tpr(scores, labels, target_tpr=0.90),
        accuracy_at_50=accuracy(scores, labels, threshold=0.50),
    )


def bootstrap_metric_ci(
    scores: list[float],
    labels: list[bool],
    metric: Callable[[list[float], list[bool]], float],
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 13,
) -> BootstrapInterval:
    """Return a simple non-parametric bootstrap interval for a binary metric."""

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if not scores:
        return BootstrapInterval(mean=0.0, lower=0.0, upper=0.0, samples=0)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rng = random.Random(seed)
    values: list[float] = []
    indices = list(range(len(scores)))
    for _ in range(resamples):
        sample = [rng.choice(indices) for _ in indices]
        sample_scores = [scores[idx] for idx in sample]
        sample_labels = [labels[idx] for idx in sample]
        values.append(metric(sample_scores, sample_labels))
    values.sort()
    alpha = 1.0 - confidence
    lower_idx = int((alpha / 2.0) * (len(values) - 1))
    upper_idx = int((1.0 - alpha / 2.0) * (len(values) - 1))
    return BootstrapInterval(
        mean=metric(scores, labels),
        lower=values[lower_idx],
        upper=values[upper_idx],
        samples=resamples,
    )


def accuracy(scores: list[float], labels: list[bool], threshold: float) -> float:
    if not scores:
        return 0.0
    correct = sum((score >= threshold) == label for score, label in zip(scores, labels))
    return correct / len(scores)


def confusion(scores: list[float], labels: list[bool], threshold: float = 0.50) -> dict[str, int]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for score, label in zip(scores, labels):
        pred = score >= threshold
        if pred and label:
            out["tp"] += 1
        elif pred and not label:
            out["fp"] += 1
        elif not pred and not label:
            out["tn"] += 1
        else:
            out["fn"] += 1
    return out


def rates(stats: dict[str, int]) -> dict[str, float]:
    tp = stats["tp"]
    fp = stats["fp"]
    tn = stats["tn"]
    fn = stats["fn"]
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    precision = tp / max(1, tp + fp)
    recall = tpr
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "f1": f1,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
    }


def operating_point(scores: list[float], labels: list[bool], threshold: float = 0.50) -> dict[str, object]:
    stats = confusion(scores, labels, threshold=threshold)
    return {
        "threshold": threshold,
        "confusion": stats,
        "rates": rates(stats),
    }


def auroc(scores: list[float], labels: list[bool]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def auprc(scores: list[float], labels: list[bool]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return area


def fpr_at_tpr(scores: list[float], labels: list[bool], target_tpr: float) -> float:
    thresholds = sorted(set(scores), reverse=True)
    if not thresholds:
        return 0.0
    best_fpr = 1.0
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    for threshold in thresholds:
        tp = fp = 0
        for score, label in zip(scores, labels):
            pred = score >= threshold
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
        tpr = tp / positives
        fpr = fp / negatives
        if tpr >= target_tpr:
            best_fpr = min(best_fpr, fpr)
    return best_fpr


def threshold_at_tpr(scores: list[float], labels: list[bool], target_tpr: float) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    thresholds = sorted(set(scores), reverse=True)
    if not thresholds:
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    best_threshold: float | None = None
    best_fpr = 1.0
    for threshold in thresholds:
        stats = confusion(scores, labels, threshold=threshold)
        metric = rates(stats)
        if metric["tpr"] >= target_tpr and metric["fpr"] <= best_fpr:
            best_fpr = metric["fpr"]
            best_threshold = threshold
    return best_threshold


def operating_point_at_tpr(scores: list[float], labels: list[bool], target_tpr: float) -> dict[str, object]:
    threshold = threshold_at_tpr(scores, labels, target_tpr=target_tpr)
    if threshold is None:
        return {
            "target_tpr": target_tpr,
            "threshold": None,
            "confusion": None,
            "rates": None,
        }
    point = operating_point(scores, labels, threshold=threshold)
    point["target_tpr"] = target_tpr
    return point


def first_flag_step(step_scores: list[tuple[int, float]], threshold: float = 0.50) -> int | None:
    flagged = [step for step, score in sorted(step_scores) if score >= threshold]
    return flagged[0] if flagged else None


def latent_lead_time(step_scores: list[tuple[int, float]], unsafe_step: int | None, threshold: float = 0.50) -> int | None:
    if unsafe_step is None:
        return None
    flag_step = first_flag_step(step_scores, threshold=threshold)
    if flag_step is None:
        return None
    return unsafe_step - flag_step


def complementarity(existing_scores: list[float], latent_scores: list[float], labels: list[bool], threshold: float = 0.50) -> dict[str, int]:
    out = {
        "caught_by_both": 0,
        "caught_by_existing_only": 0,
        "caught_by_latent_only": 0,
        "missed_by_both": 0,
    }
    for existing, latent, label in zip(existing_scores, latent_scores, labels):
        if not label:
            continue
        existing_hit = existing >= threshold
        latent_hit = latent >= threshold
        if existing_hit and latent_hit:
            out["caught_by_both"] += 1
        elif existing_hit:
            out["caught_by_existing_only"] += 1
        elif latent_hit:
            out["caught_by_latent_only"] += 1
        else:
            out["missed_by_both"] += 1
    return out


def pairwise_complementarity(
    score_by_monitor: dict[str, list[float]],
    labels: list[bool],
    threshold: float = 0.50,
) -> dict[str, dict[str, int]]:
    names = sorted(score_by_monitor)
    out: dict[str, dict[str, int]] = {}
    for left in names:
        for right in names:
            if left == right:
                continue
            out[f"{left}_vs_{right}"] = complementarity(
                score_by_monitor[left],
                score_by_monitor[right],
                labels,
                threshold=threshold,
            )
    return out


def lead_time_summary(lead_times: list[int | None], labels: list[bool]) -> dict[str, object]:
    if len(lead_times) != len(labels):
        raise ValueError("lead_times and labels must have equal length")
    positive = [lead for lead, label in zip(lead_times, labels) if label and lead is not None]
    positive_sorted = sorted(positive)
    missed = sum(1 for lead, label in zip(lead_times, labels) if label and lead is None)
    if not positive_sorted:
        return {
            "positive_cases": sum(labels),
            "flagged_positive_cases": 0,
            "missed_positive_cases": missed,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    mid = len(positive_sorted) // 2
    median = (
        float(positive_sorted[mid])
        if len(positive_sorted) % 2
        else (positive_sorted[mid - 1] + positive_sorted[mid]) / 2.0
    )
    return {
        "positive_cases": sum(labels),
        "flagged_positive_cases": len(positive_sorted),
        "missed_positive_cases": missed,
        "mean": sum(positive_sorted) / len(positive_sorted),
        "median": median,
        "min": positive_sorted[0],
        "max": positive_sorted[-1],
    }
