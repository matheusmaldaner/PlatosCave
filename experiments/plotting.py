"""experiments.plotting

Minimal plotting utilities for experiment outputs.

We avoid seaborn to keep dependencies small and deterministic.
KDE is implemented with a simple Gaussian kernel using Silverman's rule of thumb.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional


def _kde_silverman(values: List[float], grid: List[float]) -> List[float]:
    """Return kernel density estimates on `grid` for the sample `values`."""

    n = len(values)
    if n == 0:
        return [0.0 for _ in grid]
    if n == 1:
        # Degenerate: spike at the point.
        v = values[0]
        return [1.0 if abs(g - v) < 1e-12 else 0.0 for g in grid]

    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(max(var, 1e-12))
    # Silverman's rule
    h = 1.06 * std * (n ** (-1.0 / 5.0))
    h = max(h, 1e-3)

    inv = 1.0 / (n * h * math.sqrt(2.0 * math.pi))
    out = []
    for g in grid:
        s = 0.0
        for v in values:
            z = (g - v) / h
            s += math.exp(-0.5 * z * z)
        out.append(inv * s)
    return out


def save_kde_plot(
    *,
    values: List[float],
    out_path: str,
    title: str,
    x_label: str = "graph_score",
    grid_min: float = 0.0,
    grid_max: float = 1.0,
    grid_n: int = 300,
) -> Optional[str]:
    """Save a KDE plot to disk. Returns the path or None if not plotted."""

    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(clean) < 2:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    grid = [grid_min + (grid_max - grid_min) * i / (grid_n - 1) for i in range(grid_n)]
    dens = _kde_silverman(clean, grid)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(grid, dens)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(str(p), dpi=160)
    plt.close()
    return str(p)
