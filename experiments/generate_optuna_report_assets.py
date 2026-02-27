from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_IMG = REPO_ROOT / "docs" / "img"


@dataclass
class TrialRow:
    trial_number: int
    state: str
    value: float | None
    params: Dict[str, float | str]


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


def _load_trials(path: Path) -> List[TrialRow]:
    rows: List[TrialRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            params = json.loads(row.get("params_json", "") or "{}")
            rows.append(
                TrialRow(
                    trial_number=int(row["trial_number"]),
                    state=str(row["state"]),
                    value=_safe_float(row.get("value", "")),
                    params=params,
                )
            )
    return rows


def _load_study(key: str, label: str, color: str, rel_dir: str, search_space: str) -> Study:
    base = REPO_ROOT / rel_dir
    best = json.loads((base / "best_trial.json").read_text(encoding="utf-8"))
    return Study(
        key=key,
        label=label,
        color=color,
        trials_path=base / "trials.csv",
        best_path=base / "best_trial.json",
        search_space_path=REPO_ROOT / search_space,
        trials=_load_trials(base / "trials.csv"),
        best_trial_number=int(best["trial_number"]),
        best_value=float(best["value"]),
        best_params=dict(best["params"]),
    )


def _nice_ticks(vmin: float, vmax: float, approx: int = 5) -> List[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0, 1.0]
    if vmax <= vmin:
        return [vmin, vmax + 1.0]
    span = vmax - vmin
    raw = span / max(1, approx)
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm < 1.5:
        step = 1.0 * mag
    elif norm < 3.0:
        step = 2.0 * mag
    elif norm < 7.0:
        step = 5.0 * mag
    else:
        step = 10.0 * mag
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    x = start
    while x <= end + 1e-12:
        ticks.append(round(x, 10))
        x += step
    return ticks


def _svg_text(x: float, y: float, text: str, cls: str = "label", anchor: str = "start") -> str:
    return (
        f'<text class="{cls}" x="{x:.2f}" y="{y:.2f}" '
        f'text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float, cls: str = "axis") -> str:
    return f'<line class="{cls}" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" />'


def _svg_line_custom(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 2.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}"{dash_attr} />'
    )


def _svg_rect(x: float, y: float, w: float, h: float, fill: str, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="{fill}" fill-opacity="{opacity:.3f}" rx="2" />'
    )


def _svg_polyline(points: Sequence[tuple[float, float]], stroke: str, cls: str = "series") -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline class="{cls}" points="{pts}" fill="none" stroke="{stroke}" />'


def _svg_header(width: int, height: int, title: str) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(title)}</desc>',
        "<style>",
        ".bg { fill: #fbfaf5; }",
        ".panel { fill: #ffffff; stroke: #d8d4c8; stroke-width: 1; }",
        ".axis { stroke: #706b60; stroke-width: 1; }",
        ".grid { stroke: #ebe6d9; stroke-width: 1; }",
        ".series { stroke-width: 2.5; }",
        ".best { stroke: #b2382f; stroke-width: 2; stroke-dasharray: 6 4; }",
        ".mean { stroke: #7a6d2c; stroke-width: 2; stroke-dasharray: 4 3; }",
        ".title { font: 700 22px sans-serif; fill: #1d2b24; }",
        ".subtitle { font: 500 13px sans-serif; fill: #59564f; }",
        ".paneltitle { font: 700 14px sans-serif; fill: #1f2f28; }",
        ".label { font: 12px sans-serif; fill: #4e4a43; }",
        ".small { font: 11px sans-serif; fill: #6a665f; }",
        ".legend { font: 12px sans-serif; fill: #2f3b35; }",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" />',
    ]


def _save_svg(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(list(lines) + ["</svg>"]) + "\n", encoding="utf-8")


def render_histogram_panels(studies: Sequence[Study], out_path: Path) -> None:
    width, height = 1080, 900
    top = 64
    left = 72
    right = 32
    panel_gap = 28
    panel_h = 220
    panel_w = width - left - right
    all_values = [t.value for s in studies for t in s.complete if t.value is not None]
    xmin = min(all_values) - 0.01
    xmax = max(all_values) + 0.005
    bins = np.linspace(xmin, xmax, 28)
    ticks = _nice_ticks(xmin, xmax, 6)

    lines = _svg_header(width, height, "Optuna objective histograms by search stage")
    lines.append(_svg_text(left, 34, "Optuna objective distributions by stage", "title"))
    lines.append(
        _svg_text(
            left,
            54,
            "Completed-trial histograms for auc_good_vs_bad. Mean and best are marked for each stage.",
            "subtitle",
        )
    )

    for idx, study in enumerate(studies):
        y0 = top + idx * (panel_h + panel_gap)
        lines.append(f'<rect class="panel" x="{left}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="10" />')
        plot_left = left + 56
        plot_right = left + panel_w - 20
        plot_top = y0 + 24
        plot_bottom = y0 + panel_h - 36
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        values = np.array([t.value for t in study.complete if t.value is not None], dtype=float)
        counts, edges = np.histogram(values, bins=bins)
        ymax = max(1, int(counts.max()))
        yticks = _nice_ticks(0.0, float(ymax), 4)

        def x_map(v: float) -> float:
            return plot_left + (v - xmin) / (xmax - xmin) * plot_w

        def y_map(v: float) -> float:
            return plot_bottom - (v / ymax) * plot_h

        for tick in yticks:
            yy = y_map(float(tick))
            lines.append(_svg_line(plot_left, yy, plot_right, yy, "grid"))
            lines.append(_svg_text(plot_left - 8, yy + 4, f"{int(tick)}", "small", "end"))

        for tick in ticks:
            xx = x_map(float(tick))
            lines.append(_svg_line(xx, plot_bottom, xx, plot_bottom + 6, "axis"))
            lines.append(_svg_text(xx, plot_bottom + 22, f"{tick:.2f}", "small", "middle"))

        lines.append(_svg_line(plot_left, plot_top, plot_left, plot_bottom, "axis"))
        lines.append(_svg_line(plot_left, plot_bottom, plot_right, plot_bottom, "axis"))

        bar_gap = 1.5
        for count, x0, x1 in zip(counts, edges[:-1], edges[1:]):
            if count <= 0:
                continue
            xx0 = x_map(float(x0)) + bar_gap
            xx1 = x_map(float(x1)) - bar_gap
            yy = y_map(float(count))
            lines.append(_svg_rect(xx0, yy, max(1.0, xx1 - xx0), plot_bottom - yy, study.color, 0.72))

        mean_value = float(values.mean())
        best_value = float(values.max())
        lines.append(_svg_line(x_map(mean_value), plot_top, x_map(mean_value), plot_bottom, "mean"))
        lines.append(_svg_line(x_map(best_value), plot_top, x_map(best_value), plot_bottom, "best"))

        lines.append(_svg_text(plot_left, y0 + 16, study.label, "paneltitle"))
        summary = (
            f"complete {len(study.complete)}  pruned {study.pruned_count}  "
            f"best {study.best_value:.4f}  mean {mean_value:.4f}"
        )
        lines.append(_svg_text(plot_right, y0 + 16, summary, "small", "end"))
        if idx == len(studies) - 1:
            lines.append(_svg_text((plot_left + plot_right) / 2, height - 12, "auc_good_vs_bad", "label", "middle"))

    _save_svg(out_path, lines)


def render_best_so_far(studies: Sequence[Study], out_path: Path) -> None:
    width, height = 1080, 380
    left, right, top, bottom = 72, 160, 68, 46
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    xmax = max(t.trial_number for s in studies for t in s.trials)
    ymin = min(s.best_value for s in studies) - 0.01
    ymax = max(s.best_value for s in studies) + 0.003
    xticks = _nice_ticks(0.0, float(xmax), 6)
    yticks = _nice_ticks(ymin, ymax, 5)

    def x_map(v: float) -> float:
        return plot_left + (v / xmax) * plot_w

    def y_map(v: float) -> float:
        return plot_bottom - (v - ymin) / (ymax - ymin) * plot_h

    lines = _svg_header(width, height, "Optuna best-so-far AUC across search stages")
    lines.append(_svg_text(left, 34, "Best-so-far AUC across search stages", "title"))
    lines.append(
        _svg_text(
            left,
            54,
            "Cumulative best auc_good_vs_bad by trial number. Later stages improve on both score and sample efficiency.",
            "subtitle",
        )
    )
    lines.append(f'<rect class="panel" x="{left - 20}" y="{top - 20}" width="{width - left - right + 40}" height="{plot_h + 40}" rx="12" />')

    for tick in yticks:
        yy = y_map(float(tick))
        lines.append(_svg_line(plot_left, yy, plot_right, yy, "grid"))
        lines.append(_svg_text(plot_left - 8, yy + 4, f"{tick:.2f}", "small", "end"))
    for tick in xticks:
        xx = x_map(float(tick))
        lines.append(_svg_line(xx, plot_bottom, xx, plot_bottom + 6, "axis"))
        lines.append(_svg_text(xx, plot_bottom + 22, f"{int(tick)}", "small", "middle"))
    lines.append(_svg_line(plot_left, plot_top, plot_left, plot_bottom, "axis"))
    lines.append(_svg_line(plot_left, plot_bottom, plot_right, plot_bottom, "axis"))

    for idx, study in enumerate(studies):
        best = -math.inf
        points: List[tuple[float, float]] = []
        for row in sorted(study.complete, key=lambda t: t.trial_number):
            best = max(best, float(row.value))
            points.append((x_map(float(row.trial_number)), y_map(best)))
        lines.append(_svg_polyline(points, study.color))
        legend_y = plot_top + 18 + idx * 22
        lines.append(_svg_line(plot_right + 14, legend_y - 5, plot_right + 42, legend_y - 5, "series"))
        lines[-1] = lines[-1].replace('class="series"', f'class="series" stroke="{study.color}"')
        lines.append(
            _svg_text(
                plot_right + 50,
                legend_y,
                f"{study.label}  best {study.best_value:.4f}",
                "legend",
            )
        )

    lines.append(_svg_text((plot_left + plot_right) / 2, height - 12, "trial number", "label", "middle"))
    lines.append(_svg_text(18, (plot_top + plot_bottom) / 2, "best auc_good_vs_bad", "label"))
    _save_svg(out_path, lines)


def render_stage3_param_panels(stage3: Study, out_path: Path, top_n: int = 250) -> None:
    width, height = 1160, 760
    left, top = 52, 78
    cols, rows = 4, 2
    panel_w, panel_h = 250, 250
    col_gap, row_gap = 18, 22
    complete = sorted(stage3.complete, key=lambda t: float(t.value), reverse=True)[:top_n]

    specs = [
        ("graph_w.best_path", "best_path", (0.4, 0.475)),
        ("graph_w.fragility", "fragility", (-0.3, -0.25)),
        ("edge_w.synergy", "edge synergy", (0.15, 0.25)),
        ("edge_w.child_quality", "edge child_quality", (0.05, 0.2)),
        ("metric_w.method_rigor", "method_rigor", (1.5, 2.0)),
        ("metric_w.reproducibility", "reproducibility", (1.25, 2.0)),
        ("metric_w.citation_support", "citation_support", (1.5, 2.0)),
        ("penalty.eta", "penalty eta", (0.65, 0.75)),
    ]

    lines = _svg_header(width, height, "Stage-3 top-trial parameter distributions")
    lines.append(_svg_text(left, 34, "Stage-3 top-trial parameter concentrations", "title"))
    lines.append(
        _svg_text(
            left,
            54,
            f"Histograms over the top {top_n} completed stage-3 trials. Red dashed line marks the best trial value.",
            "subtitle",
        )
    )

    for idx, (key, label, (xmin, xmax)) in enumerate(specs):
        row = idx // cols
        col = idx % cols
        x0 = left + col * (panel_w + col_gap)
        y0 = top + row * (panel_h + row_gap)
        lines.append(f'<rect class="panel" x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="10" />')
        plot_left = x0 + 42
        plot_right = x0 + panel_w - 16
        plot_top = y0 + 24
        plot_bottom = y0 + panel_h - 34
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top
        values = np.array([float(t.params[key]) for t in complete], dtype=float)
        counts, edges = np.histogram(values, bins=np.linspace(xmin, xmax, 13))
        ymax = max(1, int(counts.max()))
        xticks = _nice_ticks(xmin, xmax, 4)
        yticks = _nice_ticks(0.0, float(ymax), 4)

        def x_map(v: float) -> float:
            return plot_left + (v - xmin) / (xmax - xmin) * plot_w

        def y_map(v: float) -> float:
            return plot_bottom - (v / ymax) * plot_h

        for tick in yticks:
            yy = y_map(float(tick))
            lines.append(_svg_line(plot_left, yy, plot_right, yy, "grid"))
            lines.append(_svg_text(plot_left - 6, yy + 4, f"{int(tick)}", "small", "end"))
        for tick in xticks:
            xx = x_map(float(tick))
            lines.append(_svg_line(xx, plot_bottom, xx, plot_bottom + 5, "axis"))
            fmt = f"{tick:.3f}".rstrip("0").rstrip(".")
            lines.append(_svg_text(xx, plot_bottom + 18, fmt, "small", "middle"))
        lines.append(_svg_line(plot_left, plot_top, plot_left, plot_bottom, "axis"))
        lines.append(_svg_line(plot_left, plot_bottom, plot_right, plot_bottom, "axis"))
        for count, b0, b1 in zip(counts, edges[:-1], edges[1:]):
            if count <= 0:
                continue
            xx0 = x_map(float(b0)) + 1.0
            xx1 = x_map(float(b1)) - 1.0
            yy = y_map(float(count))
            lines.append(_svg_rect(xx0, yy, max(1.0, xx1 - xx0), plot_bottom - yy, stage3.color, 0.72))
        best_value = float(stage3.best_params[key])
        lines.append(_svg_line(x_map(best_value), plot_top, x_map(best_value), plot_bottom, "best"))
        lines.append(_svg_text(plot_left, y0 + 16, label, "paneltitle"))
        lines.append(_svg_text(plot_right, y0 + 16, f"best {best_value}", "small", "end"))

    _save_svg(out_path, lines)


def _normalize_rating(label: str) -> str | None:
    raw = str(label or "").strip().lower()
    if raw.startswith("good"):
        return "Good"
    if raw.startswith("neutral"):
        return "Neutral"
    if raw.startswith("bad"):
        return "Bad"
    return None


def _best_per_paper_rows(study: Study) -> List[Dict[str, str]]:
    path = study.base_dir / "top_trial_per_paper" / f"trial_{study.best_trial_number:06d}_per_paper.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def render_score_sum_by_rating(studies: Sequence[Study], out_path: Path) -> None:
    width, height = 1120, 860
    left, top = 60, 82
    panel_w, panel_h = 1000, 220
    gap = 24
    rating_colors = {"Good": "#2f7d5b", "Neutral": "#a07c2f", "Bad": "#b64940"}
    all_vals: List[float] = []
    by_study: List[tuple[Study, Dict[str, List[float]]]] = []
    for study in studies:
        groups = {"Good": [], "Neutral": [], "Bad": []}
        for row in _best_per_paper_rows(study):
            group = _normalize_rating(row.get("rating", ""))
            if group is None:
                continue
            val = float(row["score_sum"])
            groups[group].append(val)
            all_vals.append(val)
        by_study.append((study, groups))
    xmin = min(all_vals) - 1.0
    xmax = max(all_vals) + 1.0
    bins = np.linspace(xmin, xmax, 18)
    xticks = _nice_ticks(xmin, xmax, 6)

    lines = _svg_header(width, height, "Best-trial total score distributions by paper rating")
    lines.append(_svg_text(left, 34, "Best-trial total score distributions by paper rating", "title"))
    lines.append(
        _svg_text(
            left,
            54,
            "For each stage, the best trial's per-paper raw total score (score_sum) is split by human label. "
            "Better searches push Good papers to the right and Bad papers to the left.",
            "subtitle",
        )
    )

    legend_x = width - 220
    for idx, label in enumerate(["Good", "Neutral", "Bad"]):
        yy = 34 + idx * 18
        lines.append(_svg_rect(legend_x, yy - 10, 14, 10, rating_colors[label], 0.78))
        lines.append(_svg_text(legend_x + 22, yy, label, "legend"))

    for idx, (study, groups) in enumerate(by_study):
        y0 = top + idx * (panel_h + gap)
        lines.append(f'<rect class="panel" x="{left}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="10" />')
        plot_left = left + 56
        plot_right = left + panel_w - 20
        plot_top = y0 + 24
        plot_bottom = y0 + panel_h - 36
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top
        ymax = 1
        histograms = {}
        for label, vals in groups.items():
            counts, edges = np.histogram(np.array(vals, dtype=float), bins=bins)
            histograms[label] = (counts, edges)
            ymax = max(ymax, int(counts.max()))
        yticks = _nice_ticks(0.0, float(ymax), 4)

        def x_map(v: float) -> float:
            return plot_left + (v - xmin) / (xmax - xmin) * plot_w

        def y_map(v: float) -> float:
            return plot_bottom - (v / ymax) * plot_h

        for tick in yticks:
            yy = y_map(float(tick))
            lines.append(_svg_line(plot_left, yy, plot_right, yy, "grid"))
            lines.append(_svg_text(plot_left - 8, yy + 4, f"{int(tick)}", "small", "end"))
        for tick in xticks:
            xx = x_map(float(tick))
            lines.append(_svg_line(xx, plot_bottom, xx, plot_bottom + 6, "axis"))
            lines.append(_svg_text(xx, plot_bottom + 22, f"{tick:.0f}", "small", "middle"))
        lines.append(_svg_line(plot_left, plot_top, plot_left, plot_bottom, "axis"))
        lines.append(_svg_line(plot_left, plot_bottom, plot_right, plot_bottom, "axis"))

        # Draw three semi-transparent histograms on the same axes.
        offsets = {"Good": -0.22, "Neutral": 0.0, "Bad": 0.22}
        for label in ["Good", "Neutral", "Bad"]:
            counts, edges = histograms[label]
            for count, b0, b1 in zip(counts, edges[:-1], edges[1:]):
                if count <= 0:
                    continue
                bin_w = x_map(float(b1)) - x_map(float(b0))
                xx0 = x_map(float(b0)) + bin_w * (0.18 + offsets[label] * 0.25)
                bar_w = max(1.5, bin_w * 0.22)
                yy = y_map(float(count))
                lines.append(_svg_rect(xx0, yy, bar_w, plot_bottom - yy, rating_colors[label], 0.68))

        means = []
        for label in ["Good", "Neutral", "Bad"]:
            vals = groups[label]
            if vals:
                mean_val = float(np.mean(vals))
                means.append(f"{label} mean {mean_val:.1f}")
                lines.append(
                    _svg_line_custom(
                        x_map(mean_val),
                        plot_top,
                        x_map(mean_val),
                        plot_bottom,
                        stroke=rating_colors[label],
                        width=2.0,
                        dash="6 4",
                    )
                )

        lines.append(_svg_text(plot_left, y0 + 16, study.label, "paneltitle"))
        counts_str = "  ".join(f"{label} {len(groups[label])}" for label in ["Good", "Neutral", "Bad"])
        lines.append(_svg_text(plot_right, y0 + 16, counts_str, "small", "end"))
        lines.append(_svg_text(plot_left, plot_top + 14, "  ".join(means), "small"))

    lines.append(_svg_text(width / 2, height - 12, "score_sum for the best trial of each stage", "label", "middle"))
    _save_svg(out_path, lines)


def render_summary_json(studies: Sequence[Study], out_path: Path) -> None:
    out = []
    for study in studies:
        out.append(
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
            }
        )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
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
    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    render_histogram_panels(studies, DOCS_IMG / "optuna_stage_objective_histograms.svg")
    render_best_so_far(studies, DOCS_IMG / "optuna_stage_best_so_far.svg")
    render_stage3_param_panels(studies[-1], DOCS_IMG / "optuna_stage3_key_param_histograms.svg")
    render_score_sum_by_rating(studies, DOCS_IMG / "optuna_stage_score_sum_by_rating.svg")
    render_summary_json(studies, DOCS_IMG / "optuna_stage_summary.json")


if __name__ == "__main__":
    main()
