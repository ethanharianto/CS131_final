"""Team-purity metric for per-track team-id predictions vs hand-labeled GT."""

from __future__ import annotations

from collections import Counter


def team_purity(
    predicted: dict[int, str], gt: dict[int, str]
) -> dict[str, float | int | dict]:
    """Per-track team purity.

    predicted: {predicted_track_id -> "A" | "B" | "other"}
    gt:        {predicted_track_id -> "A" | "B" | "other"}  (subset; only labeled tracks)

    Returns overall accuracy on labeled tracks, plus per-class precision/recall and a
    confusion matrix. Predicted tracks without a GT label are skipped (not penalized).
    """
    labels = ("A", "B", "other")
    matrix = {g: {p: 0 for p in labels} for g in labels}
    n_labeled = 0
    n_correct = 0

    for tid, g in gt.items():
        if g not in labels:
            continue
        p = predicted.get(tid)
        if p not in labels:
            continue
        matrix[g][p] += 1
        n_labeled += 1
        if g == p:
            n_correct += 1

    overall = n_correct / n_labeled if n_labeled else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    for cls in labels:
        tp = matrix[cls][cls]
        fp = sum(matrix[g][cls] for g in labels if g != cls)
        fn = sum(matrix[cls][p] for p in labels if p != cls)
        support = sum(matrix[cls].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[cls] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    return {
        "n_labeled_tracks": n_labeled,
        "n_correct": n_correct,
        "overall_purity": round(overall, 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def gt_team_distribution(gt: dict[int, str]) -> dict[str, int]:
    return dict(Counter(gt.values()))


def bootstrap_purity_ci(
    predicted: dict[int, str],
    gt: dict[int, str],
    *,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Stratified bootstrap CI for overall team purity on labeled tracks."""
    import random

    rng = random.Random(seed)
    labeled = [(tid, g) for tid, g in gt.items() if g in ("A", "B", "other")]
    if len(labeled) < 2:
        p = team_purity(predicted, gt)["overall_purity"]
        return {"overall_purity": p, "ci_low": p, "ci_high": p, "n_bootstrap": 0}

    by_class: dict[str, list[tuple[int, str]]] = {"A": [], "B": [], "other": []}
    for tid, g in labeled:
        by_class[g].append((tid, g))

    purities: list[float] = []
    for _ in range(n_bootstrap):
        sample: list[tuple[int, str]] = []
        for cls in by_class:
            pool = by_class[cls]
            if not pool:
                continue
            sample.extend(rng.choices(pool, k=len(pool)))
        gt_s = {tid: g for tid, g in sample}
        purities.append(team_purity(predicted, gt_s)["overall_purity"])

    purities.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1.0 - alpha) * n_bootstrap) - 1
    point = team_purity(predicted, gt)["overall_purity"]
    return {
        "overall_purity": point,
        "ci_low": round(purities[lo_idx], 4),
        "ci_high": round(purities[hi_idx], 4),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }
