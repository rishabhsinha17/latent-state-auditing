from __future__ import annotations


def select_threshold(
    scores: list[float],
    labels: list[bool],
    objective: str = "youden",
    max_fpr: float | None = None,
) -> float:
    """Select a decision threshold on a calibration split.

    Objectives:
    - `youden`: maximize TPR - FPR.
    - `f1`: maximize binary F1.
    - `accuracy`: maximize accuracy.
    - `max_tpr_at_fpr`: maximize TPR subject to `max_fpr`.
    """

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if not scores:
        raise ValueError("cannot calibrate on empty scores")
    thresholds = sorted(set(scores), reverse=True)
    thresholds = [thresholds[0] + 1e-12] + thresholds + [thresholds[-1] - 1e-12]
    best_threshold = thresholds[0]
    best_value = float("-inf")
    for threshold in thresholds:
        stats = confusion(scores, labels, threshold)
        value = _objective_value(stats, objective=objective, max_fpr=max_fpr)
        if value > best_value:
            best_value = value
            best_threshold = threshold
    return best_threshold


def confusion(scores: list[float], labels: list[bool], threshold: float) -> dict[str, int]:
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
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
    }


def _objective_value(stats: dict[str, int], objective: str, max_fpr: float | None) -> float:
    metric = rates(stats)
    if objective == "youden":
        return metric["tpr"] - metric["fpr"]
    if objective == "f1":
        return metric["f1"]
    if objective == "accuracy":
        return metric["accuracy"]
    if objective == "max_tpr_at_fpr":
        if max_fpr is None:
            raise ValueError("max_fpr_at_fpr requires max_fpr")
        if metric["fpr"] > max_fpr:
            return float("-inf")
        return metric["tpr"]
    raise ValueError(f"unknown threshold objective: {objective}")
