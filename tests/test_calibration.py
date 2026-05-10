from latent_agent_auditing.eval.calibration import confusion, rates, select_threshold


def test_select_threshold_separates_simple_scores() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [True, True, False, False]
    threshold = select_threshold(scores, labels)
    stats = confusion(scores, labels, threshold)
    assert stats == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_select_threshold_with_fpr_constraint() -> None:
    scores = [0.9, 0.7, 0.6, 0.55, 0.1]
    labels = [True, True, False, False, False]
    threshold = select_threshold(scores, labels, objective="max_tpr_at_fpr", max_fpr=0.0)
    stats = confusion(scores, labels, threshold)
    assert stats["fp"] == 0
    assert stats["tp"] >= 1


def test_rates() -> None:
    metric = rates({"tp": 3, "fp": 1, "tn": 4, "fn": 2})
    assert metric["tpr"] == 0.6
    assert metric["fpr"] == 0.2
