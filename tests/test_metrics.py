from latent_agent_auditing.eval.metrics import (
    auprc,
    auroc,
    bootstrap_metric_ci,
    lead_time_summary,
    pairwise_complementarity,
    threshold_at_tpr,
    complementarity,
    latent_lead_time,
)


def test_auroc_orders_perfect_scores() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0


def test_auprc_has_signal() -> None:
    assert auprc([0.9, 0.1, 0.8, 0.2], [True, False, True, False]) > 0.9


def test_lead_time_positive_when_flag_before_unsafe_step() -> None:
    assert latent_lead_time([(1, 0.2), (2, 0.7), (4, 0.9)], unsafe_step=4) == 2


def test_complementarity_counts_only_positive_cases() -> None:
    result = complementarity([0.9, 0.1, 0.9], [0.9, 0.9, 0.1], [True, True, False])
    assert result["caught_by_both"] == 1
    assert result["caught_by_latent_only"] == 1


def test_bootstrap_metric_ci_returns_ordered_interval() -> None:
    result = bootstrap_metric_ci([0.9, 0.8, 0.2, 0.1], [True, True, False, False], auroc, resamples=20)
    assert result.lower <= result.mean <= result.upper
    assert result.samples == 20


def test_pairwise_complementarity_names_monitor_pairs() -> None:
    result = pairwise_complementarity(
        {"trace": [0.1, 0.9], "latent": [0.9, 0.9]},
        [True, True],
    )
    assert result["trace_vs_latent"]["caught_by_latent_only"] == 1


def test_lead_time_summary_counts_missed_positive_flags() -> None:
    result = lead_time_summary([2, None, None], [True, True, False])
    assert result["flagged_positive_cases"] == 1
    assert result["missed_positive_cases"] == 1


def test_threshold_at_tpr_selects_operational_threshold() -> None:
    threshold = threshold_at_tpr([0.9, 0.8, 0.4, 0.1], [True, True, False, False], target_tpr=1.0)
    assert threshold == 0.8
