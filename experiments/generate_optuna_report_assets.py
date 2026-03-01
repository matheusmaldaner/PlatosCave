from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "Matplotlib is required for report asset generation. Install it in "
        "the experiment environment, e.g. '.venv-exp/bin/pip install matplotlib'."
    ) from exc

try:
    from plotting import _kde_silverman
except Exception:
    from .plotting import _kde_silverman


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_IMG = REPO_ROOT / "docs" / "img"
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 0

RATING_ORDER = ("Good", "Neutral", "Bad")
RATING_COLORS = {
    "Good": "#287d6e",
    "Neutral": "#b9871f",
    "Bad": "#b54f4f",
}
COMPARISON_COLORS = {
    "dense_to_refine": "#cc6b2c",
    "refine_to_stage3": "#2d7c67",
}


def _configure_matplotlib() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "#fbfaf7",
            "axes.facecolor": "#f7f4ed",
            "axes.edgecolor": "#bcb5aa",
            "axes.labelcolor": "#2f2b26",
            "axes.titlecolor": "#211d18",
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "grid.color": "#ddd6ca",
            "grid.alpha": 0.8,
            "grid.linewidth": 0.8,
            "legend.facecolor": "#fffdf9",
            "legend.edgecolor": "#cdc5b8",
            "legend.framealpha": 0.96,
            "legend.fancybox": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 240,
            "savefig.facecolor": "#fbfaf7",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


@dataclass(frozen=True)
class TrialRow:
    trial_number: int
    state: str
    value: float | None
    params: Dict[str, float | str]


@dataclass(frozen=True)
class PaperScore:
    paper_key: str
    raw_rating: str
    grouped_rating: str
    graph_score_mean: float
    score_sum: float


@dataclass
class Study:
    key: str
    label: str
    color: str
    trials_path: Path
    best_path: Path
    search_space_path: Path
    trials: List[TrialRow]
    best_trial_number: int
    best_value: float
    best_params: Dict[str, float | str]
    best_papers: Dict[str, PaperScore]

    @property
    def complete(self) -> List[TrialRow]:
        return [t for t in self.trials if t.state == "COMPLETE" and t.value is not None]

    @property
    def pruned_count(self) -> int:
        return sum(t.state == "PRUNED" for t in self.trials)

    @property
    def base_dir(self) -> Path:
        return self.trials_path.parent


def _safe_float(text: str) -> float | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return None
    return float(text)


def _normalize_rating_group(label: str) -> str | None:
    raw = str(label or "").strip().lower()
    if raw.startswith("good"):
        return "Good"
    if raw.startswith("neutral"):
        return "Neutral"
    if raw.startswith("bad"):
        return "Bad"
    return None


def _rating_code(label: str) -> int:
    if label == "Good":
        return 1
    if label == "Neutral":
        return 0
    if label == "Bad":
        return -1
    raise ValueError(f"Unsupported rating {label!r}")


def _objective_rating_code(label: str) -> int | None:
    raw = str(label or "").strip().lower()
    if raw == "good":
        return 1
    if raw == "neutral":
        return 0
    if raw == "bad":
        return -1
    return None


def _load_trials(path: Path) -> List[TrialRow]:
    rows: List[TrialRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                TrialRow(
                    trial_number=int(row["trial_number"]),
                    state=str(row["state"]),
                    value=_safe_float(row.get("value", "")),
                    params=json.loads(row.get("params_json", "") or "{}"),
                )
            )
    return rows


def _best_per_paper_path(base_dir: Path, trial_number: int) -> Path:
    return base_dir / "top_trial_per_paper" / f"trial_{trial_number:06d}_per_paper.csv"


def _load_best_papers(path: Path) -> Dict[str, PaperScore]:
    rows: Dict[str, PaperScore] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_rating = str(row.get("rating", ""))
            grouped_rating = _normalize_rating_group(raw_rating)
            if grouped_rating is None:
                continue
            key = str(row["paper_key"])
            rows[key] = PaperScore(
                paper_key=key,
                raw_rating=raw_rating,
                grouped_rating=grouped_rating,
                graph_score_mean=float(row["graph_score_mean"]),
                score_sum=float(row["score_sum"]),
            )
    return rows


def _load_study(key: str, label: str, color: str, rel_dir: str, search_space: str) -> Study:
    base = REPO_ROOT / rel_dir
    best = json.loads((base / "best_trial.json").read_text(encoding="utf-8"))
    best_trial_number = int(best["trial_number"])
    return Study(
        key=key,
        label=label,
        color=color,
        trials_path=base / "trials.csv",
        best_path=base / "best_trial.json",
        search_space_path=REPO_ROOT / search_space,
        trials=_load_trials(base / "trials.csv"),
        best_trial_number=best_trial_number,
        best_value=float(best["value"]),
        best_params=dict(best["params"]),
        best_papers=_load_best_papers(_best_per_paper_path(base, best_trial_number)),
    )


def _mix_color(color: str, toward: str = "#ffffff", alpha: float = 0.3) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color), dtype=float)
    dest = np.array(mcolors.to_rgb(toward), dtype=float)
    mixed = (1.0 - alpha) * base + alpha * dest
    return tuple(float(x) for x in mixed)


def _kde_points(values: np.ndarray, xmin: float, xmax: float, grid_n: int = 320) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([xmin, xmax], dtype=float), np.zeros(2, dtype=float)
    grid = np.linspace(xmin, xmax, grid_n)
    density = np.array(_kde_silverman(values.tolist(), grid.tolist()), dtype=float)
    return grid, density


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = math.sqrt(float(np.dot(xc, xc)) * float(np.dot(yc, yc)))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def _spearman_rating_score(ratings: np.ndarray, scores: np.ndarray) -> float:
    return _pearson_corr(_rankdata_average(ratings.astype(float)), _rankdata_average(scores.astype(float)))


def _auc_good_vs_bad(ratings: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[ratings == 1]
    neg = scores[ratings == -1]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUC requires at least one Good and one Bad paper.")
    joined = np.concatenate([pos, neg])
    ranks = _rankdata_average(joined)
    n_pos = pos.size
    n_neg = neg.size
    rank_sum_pos = float(np.sum(ranks[:n_pos]))
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def _study_scores(study: Study) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(study.best_papers)
    ratings = np.array([_rating_code(study.best_papers[key].grouped_rating) for key in keys], dtype=int)
    graph_scores = np.array([study.best_papers[key].graph_score_mean for key in keys], dtype=float)
    score_sums = np.array([study.best_papers[key].score_sum for key in keys], dtype=float)
    return ratings, graph_scores, score_sums


def _study_metrics(study: Study) -> Dict[str, object]:
    grouped_ratings, graph_scores, score_sums = _study_scores(study)
    keys = sorted(study.best_papers)
    objective_codes = np.array(
        [_objective_rating_code(study.best_papers[key].raw_rating) for key in keys],
        dtype=object,
    )
    objective_mask = np.array([code is not None for code in objective_codes], dtype=bool)
    objective_ratings = np.array([int(code) for code in objective_codes[objective_mask]], dtype=int)
    objective_scores = graph_scores[objective_mask]

    grouped_rating_means = {
        label: float(np.mean(graph_scores[grouped_ratings == _rating_code(label)]))
        for label in RATING_ORDER
    }
    score_sum_means = {
        label: float(np.mean(score_sums[grouped_ratings == _rating_code(label)]))
        for label in RATING_ORDER
    }
    grouped_rating_counts = {
        label: int(np.sum(grouped_ratings == _rating_code(label)))
        for label in RATING_ORDER
    }
    objective_rating_counts = {
        "Good": int(np.sum(objective_ratings == 1)),
        "Neutral": int(np.sum(objective_ratings == 0)),
        "Bad": int(np.sum(objective_ratings == -1)),
    }
    return {
        "auc_good_vs_bad": _auc_good_vs_bad(objective_ratings, objective_scores),
        "spearman_good_neutral_bad": _spearman_rating_score(objective_ratings, objective_scores),
        "graph_score_mean": float(np.mean(graph_scores)),
        "grouped_good_bad_gap": grouped_rating_means["Good"] - grouped_rating_means["Bad"],
        "grouped_good_neutral_gap": grouped_rating_means["Good"] - grouped_rating_means["Neutral"],
        "objective_papers": int(objective_ratings.size),
        "grouped_rating_counts": grouped_rating_counts,
        "objective_rating_counts": objective_rating_counts,
        "grouped_rating_means": grouped_rating_means,
        "score_sum_means": score_sum_means,
    }


def _aligned_stage_scores(base: Study, candidate: Study) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(set(base.best_papers) & set(candidate.best_papers))
    if not keys:
        raise ValueError(f"No shared papers between {base.label} and {candidate.label}.")
    base_objective = np.array([_objective_rating_code(base.best_papers[key].raw_rating) for key in keys], dtype=object)
    cand_objective = np.array([_objective_rating_code(candidate.best_papers[key].raw_rating) for key in keys], dtype=object)
    if not np.array_equal(base_objective, cand_objective):
        raise ValueError(f"Rating mismatch between {base.label} and {candidate.label}.")
    base_scores = np.array([base.best_papers[key].graph_score_mean for key in keys], dtype=float)
    cand_scores = np.array([candidate.best_papers[key].graph_score_mean for key in keys], dtype=float)
    return base_objective, base_scores, cand_scores


def _summarize_bootstrap(samples: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "q05": float(np.quantile(samples, 0.05)),
        "q50": float(np.quantile(samples, 0.50)),
        "q95": float(np.quantile(samples, 0.95)),
        "p_gt_0": float(np.mean(samples > 0.0)),
    }


def _bootstrap_delta(base: Study, candidate: Study) -> Dict[str, object]:
    objective_codes, base_scores, cand_scores = _aligned_stage_scores(base, candidate)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    spearman = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    mean_score = np.empty(BOOTSTRAP_SAMPLES, dtype=float)

    filled = 0
    while filled < BOOTSTRAP_SAMPLES:
        idx = rng.integers(0, base_scores.size, size=base_scores.size)
        sub_objective = objective_codes[idx]
        objective_mask = np.array([code is not None for code in sub_objective], dtype=bool)
        sub_ratings = np.array([int(code) for code in sub_objective[objective_mask]], dtype=int)
        if np.count_nonzero(sub_ratings == 1) == 0 or np.count_nonzero(sub_ratings == -1) == 0:
            continue
        sub_base = base_scores[idx]
        sub_cand = cand_scores[idx]
        auc[filled] = _auc_good_vs_bad(sub_ratings, sub_cand[objective_mask]) - _auc_good_vs_bad(
            sub_ratings, sub_base[objective_mask]
        )
        spearman[filled] = _spearman_rating_score(sub_ratings, sub_cand[objective_mask]) - _spearman_rating_score(
            sub_ratings, sub_base[objective_mask]
        )
        mean_score[filled] = float(np.mean(sub_cand) - np.mean(sub_base))
        filled += 1

    comp_key = f"{base.key}_to_{candidate.key}"
    samples = {
        "auc_good_vs_bad": auc,
        "spearman_good_neutral_bad": spearman,
        "graph_score_mean": mean_score,
    }
    return {
        "key": comp_key,
        "label": f"{base.label} -> {candidate.label}",
        "from": base.key,
        "to": candidate.key,
        "n_papers": int(base_scores.size),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "color": COMPARISON_COLORS[comp_key],
        "samples": samples,
        "summary": {metric: _summarize_bootstrap(vals) for metric, vals in samples.items()},
    }


def _save_figure(fig, stem: str) -> List[str]:
    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for ext in ("png", "pdf", "svg"):
        path = DOCS_IMG / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 260
        fig.savefig(path, **kwargs)
        saved.append(str(path.relative_to(REPO_ROOT)))
    plt.close(fig)
    return saved


def render_objective_histograms(studies: Sequence[Study]) -> List[str]:
    fig, axes = plt.subplots(len(studies), 1, figsize=(11.5, 10.2), sharex=True, constrained_layout=True)
    if len(studies) == 1:
        axes = [axes]
    all_values = np.concatenate([np.array([t.value for t in s.complete], dtype=float) for s in studies])
    xmin = float(np.min(all_values)) - 0.01
    xmax = float(np.max(all_values)) + 0.008
    bins = np.linspace(xmin, xmax, 28)

    for ax, study in zip(axes, studies):
        values = np.array([t.value for t in study.complete], dtype=float)
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=_mix_color(study.color, alpha=0.16),
            edgecolor=_mix_color(study.color, toward="#2b241d", alpha=0.05),
            linewidth=1.2,
            label=f"Completed trials (n={values.size})",
        )
        grid, density = _kde_points(values, xmin, xmax)
        ax.plot(grid, density, color=study.color, linewidth=2.4, label="KDE")
        mean_value = float(np.mean(values))
        ax.axvline(mean_value, color="#6f5c2b", linestyle="--", linewidth=2.0, label=f"Mean = {mean_value:.4f}")
        ax.axvline(
            study.best_value,
            color="#b23b3b",
            linestyle="-.",
            linewidth=2.0,
            label=f"Best = {study.best_value:.4f}",
        )
        ax.set_title(f"{study.label}: complete {len(study.complete)}, pruned {study.pruned_count}")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", ncol=2, fontsize=9)

    axes[-1].set_xlabel("auc_good_vs_bad")
    fig.suptitle("Optuna objective distributions by search stage", fontsize=17, fontweight="semibold")
    return _save_figure(fig, "optuna_stage_objective_histograms")


def render_best_so_far(studies: Sequence[Study]) -> List[str]:
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    for study in studies:
        rows = sorted(study.complete, key=lambda row: row.trial_number)
        x = np.array([row.trial_number for row in rows], dtype=float)
        y = np.maximum.accumulate(np.array([row.value for row in rows], dtype=float))
        ax.step(x, y, where="post", linewidth=2.4, color=study.color, label=f"{study.label} (best {study.best_value:.4f})")
        ax.scatter(
            [study.best_trial_number],
            [study.best_value],
            color=study.color,
            edgecolors="#201a15",
            linewidths=0.8,
            marker="o",
            s=55,
            zorder=3,
        )

    ax.set_title("Best-so-far auc_good_vs_bad across the staged searches")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best auc_good_vs_bad so far")
    ax.legend(loc="lower right")
    return _save_figure(fig, "optuna_stage_best_so_far")


def render_stage3_param_histograms(stage3: Study, top_n: int = 250) -> List[str]:
    specs = [
        ("graph_w.best_path", "graph_w.best_path"),
        ("graph_w.fragility", "graph_w.fragility"),
        ("edge_w.synergy", "edge_w.synergy"),
        ("edge_w.child_quality", "edge_w.child_quality"),
        ("metric_w.method_rigor", "metric_w.method_rigor"),
        ("metric_w.reproducibility", "metric_w.reproducibility"),
        ("metric_w.citation_support", "metric_w.citation_support"),
        ("penalty.eta", "penalty.eta"),
    ]
    top_trials = sorted(stage3.complete, key=lambda row: float(row.value), reverse=True)[:top_n]
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.6), constrained_layout=True)

    for ax, (param_key, label) in zip(axes.flat, specs):
        values = np.array([float(row.params[param_key]) for row in top_trials], dtype=float)
        unique = np.unique(values)
        if unique.size == 1:
            step = 1.0
            bins = np.array([unique[0] - 0.5, unique[0] + 0.5], dtype=float)
        else:
            step = float(np.min(np.diff(unique)))
            bins = np.arange(unique.min() - step / 2.0, unique.max() + step, step)
        ax.hist(
            values,
            bins=bins,
            color=_mix_color(stage3.color, alpha=0.18),
            edgecolor=_mix_color(stage3.color, toward="#1c1711", alpha=0.05),
            linewidth=1.1,
            label=f"Top {top_n} trials",
        )
        ax.axvline(
            float(np.median(values)),
            color="#6f5c2b",
            linestyle=":",
            linewidth=1.8,
            label=f"Median = {float(np.median(values)):.4g}",
        )
        ax.axvline(
            float(stage3.best_params[param_key]),
            color="#b23b3b",
            linestyle="--",
            linewidth=2.0,
            label=f"Best = {float(stage3.best_params[param_key]):.4g}",
        )
        ax.set_title(label)
        ax.set_ylabel("Count")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Stage-3 parameter concentrations in the top 250 completed trials", fontsize=17, fontweight="semibold")
    return _save_figure(fig, "optuna_stage3_key_param_histograms")


def render_score_sum_by_rating(studies: Sequence[Study]) -> List[str]:
    fig, axes = plt.subplots(len(studies), 1, figsize=(11.5, 10.2), sharex=True, constrained_layout=True)
    if len(studies) == 1:
        axes = [axes]
    all_values = []
    for study in studies:
        for paper in study.best_papers.values():
            all_values.append(paper.score_sum)
    xmin = float(min(all_values)) - 1.0
    xmax = float(max(all_values)) + 1.0
    bins = np.linspace(xmin, xmax, 18)

    for ax, study in zip(axes, studies):
        for rating in RATING_ORDER:
            values = np.array(
                [paper.score_sum for paper in study.best_papers.values() if paper.grouped_rating == rating],
                dtype=float,
            )
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.22,
                color=RATING_COLORS[rating],
                edgecolor=RATING_COLORS[rating],
                linewidth=1.1,
                label=f"{rating} (mean {float(np.mean(values)):.1f}, n={values.size})",
            )
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color=RATING_COLORS[rating],
                linewidth=1.5,
            )
            ax.axvline(float(np.mean(values)), color=RATING_COLORS[rating], linestyle="--", linewidth=1.8)
        ax.set_title(f"{study.label}: best-trial score_sum by human rating")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", fontsize=9)

    axes[-1].set_xlabel("score_sum")
    fig.suptitle(
        "Best-trial raw total score distributions by rating",
        fontsize=17,
        fontweight="semibold",
    )
    return _save_figure(fig, "optuna_stage_score_sum_by_rating")


def render_bootstrap_deltas(comparisons: Sequence[Dict[str, object]]) -> List[str]:
    metrics = [
        ("auc_good_vs_bad", "Bootstrap delta: auc_good_vs_bad"),
        ("spearman_good_neutral_bad", "Bootstrap delta: Spearman"),
        ("graph_score_mean", "Bootstrap delta: mean graph_score_mean"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.1), constrained_layout=True)

    for ax, (metric_key, title) in zip(axes, metrics):
        all_samples = np.concatenate(
            [np.asarray(comp["samples"][metric_key], dtype=float) for comp in comparisons]
        )
        span = float(np.max(all_samples) - np.min(all_samples))
        pad = max(span * 0.08, 1e-3)
        xmin = float(np.min(all_samples)) - pad
        xmax = float(np.max(all_samples)) + pad
        bins = np.linspace(xmin, xmax, 32)

        for comp in comparisons:
            samples = np.asarray(comp["samples"][metric_key], dtype=float)
            summary = comp["summary"][metric_key]
            color = str(comp["color"])
            ax.hist(
                samples,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.22,
                color=color,
                edgecolor=color,
                linewidth=1.1,
                label=(
                    f"{comp['label']} "
                    f"(median {summary['q50']:+.4f}, 90% [{summary['q05']:+.4f}, {summary['q95']:+.4f}])"
                ),
            )
            grid, density = _kde_points(samples, xmin, xmax)
            ax.plot(grid, density, color=color, linewidth=2.2)
            ax.axvline(float(summary["q50"]), color=color, linestyle="--", linewidth=1.8)

        ax.axvline(0.0, color="#3b342c", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Candidate stage minus previous stage")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Bootstrap evidence for robust stage-2 gains versus stage-3 objective specialization",
        fontsize=17,
        fontweight="semibold",
    )
    return _save_figure(fig, "optuna_stage_bootstrap_deltas")


def render_summary_json(
    studies: Sequence[Study],
    stage_metrics: Dict[str, Dict[str, object]],
    comparisons: Sequence[Dict[str, object]],
    assets: Dict[str, List[str]],
) -> None:
    out = {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "stages": [],
        "comparisons": [],
        "assets": assets,
    }
    for study in studies:
        metrics = stage_metrics[study.key]
        out["stages"].append(
            {
                "study": study.key,
                "label": study.label,
                "trials_total": len(study.trials),
                "trials_complete": len(study.complete),
                "trials_pruned": study.pruned_count,
                "best_trial_number": study.best_trial_number,
                "best_value": study.best_value,
                "search_space_path": str(study.search_space_path.relative_to(REPO_ROOT)),
                "trials_path": str(study.trials_path.relative_to(REPO_ROOT)),
                "best_trial_path": str(study.best_path.relative_to(REPO_ROOT)),
                "metrics": metrics,
            }
        )
    for comp in comparisons:
        out["comparisons"].append(
            {
                "key": comp["key"],
                "label": comp["label"],
                "from": comp["from"],
                "to": comp["to"],
                "n_papers": comp["n_papers"],
                "bootstrap_samples": comp["bootstrap_samples"],
                "summary": comp["summary"],
            }
        )
    (DOCS_IMG / "optuna_stage_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
    _configure_matplotlib()
    studies = [
        _load_study(
            key="dense",
            label="Stage 1 dense",
            color="#355c7d",
            rel_dir="runs/optuna_search_live/optuna_debug4_dense_v1",
            search_space="experiments/search_spaces/optuna_debug4_dense_v1.json",
        ),
        _load_study(
            key="refine",
            label="Stage 2 refine",
            color="#c06c3e",
            rel_dir="runs/optuna_search_refine/optuna_debug4_refine_v1",
            search_space="experiments/search_spaces/optuna_debug4_refine_v1.json",
        ),
        _load_study(
            key="stage3",
            label="Stage 3 sparse",
            color="#2f7d5b",
            rel_dir="runs/optuna_search_stage3/optuna_debug4_stage3_sparse_v1",
            search_space="experiments/search_spaces/optuna_debug4_stage3_sparse_v1.json",
        ),
    ]
    stage_metrics = {study.key: _study_metrics(study) for study in studies}
    comparisons = [
        _bootstrap_delta(studies[0], studies[1]),
        _bootstrap_delta(studies[1], studies[2]),
    ]

    assets = {
        "optuna_stage_objective_histograms": render_objective_histograms(studies),
        "optuna_stage_best_so_far": render_best_so_far(studies),
        "optuna_stage_score_sum_by_rating": render_score_sum_by_rating(studies),
        "optuna_stage3_key_param_histograms": render_stage3_param_histograms(studies[-1]),
        "optuna_stage_bootstrap_deltas": render_bootstrap_deltas(comparisons),
    }
    render_summary_json(studies, stage_metrics, comparisons, assets)


if __name__ == "__main__":
    main()
