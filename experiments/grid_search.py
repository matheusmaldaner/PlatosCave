from __future__ import annotations

import csv
import copy
import hashlib
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from ablation_studies import (
        AblationError,
        _auc_good_bad,
        _build_kg_from_dag,
        _csv_read_rows,
        _csv_write,
        _discover_papers,
        _filter_papers,
        _import_scorer,
        _json_dump,
        _load_trial_inputs,
        _node_results_copy,
        _safe_float,
        _spearman,
        _json_load,
    )
except Exception:
    from .ablation_studies import (
        AblationError,
        _auc_good_bad,
        _build_kg_from_dag,
        _csv_read_rows,
        _csv_write,
        _discover_papers,
        _filter_papers,
        _import_scorer,
        _json_dump,
        _load_trial_inputs,
        _node_results_copy,
        _safe_float,
        _spearman,
        _json_load,
    )


METRICS: Tuple[str, ...] = (
    "credibility",
    "relevance",
    "evidence_strength",
    "method_rigor",
    "reproducibility",
    "citation_support",
)

SUPPORTED_PARAM_PATTERNS: Dict[str, str] = {
    "metric_w.<metric>": "Global node-metric weight override. Metrics: "
    + ",".join(METRICS),
    "node_q.<Role>.<metric>": "Role-specific node-quality weight override.",
    "edge_w.<feature>": "Edge combine weights: role_prior,parent_quality,child_quality,alignment,synergy",
    "graph_w.<component>": "Graph score weights: bridge_coverage,best_path,redundancy,fragility,coherence,coverage",
    "penalty.<field>": "Propagation penalty fields: enabled,alpha,eta,default_raw_conf,softmin_beta,dampmin_lambda",
}


def _safe_col(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _load_search_space(path: str) -> Dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return _normalize_search_space(obj)


def _normalize_search_space(space: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(space, dict):
        raise AblationError("Search space must be a JSON object.")
    params = space.get("params")
    if not isinstance(params, dict) or not params:
        raise AblationError("Search space must define a non-empty 'params' object.")

    out_params: Dict[str, List[Any]] = {}
    for key, values in params.items():
        if not isinstance(key, str) or not key.strip():
            raise AblationError("Every search-space param key must be a non-empty string.")
        if not isinstance(values, list) or not values:
            raise AblationError(f"Param '{key}' must map to a non-empty JSON array.")
        out_params[key.strip()] = list(values)

    constraints = space.get("constraints", [])
    if constraints is None:
        constraints = []
    if not isinstance(constraints, list):
        raise AblationError("'constraints' must be a list when provided.")

    normalized = {
        "name": str(space.get("name", "numeric_grid_search")).strip() or "numeric_grid_search",
        "description": str(space.get("description", "") or "").strip(),
        "params": out_params,
        "constraints": constraints,
        "normalize_metric_weights": bool(space.get("normalize_metric_weights", False)),
        "include_component_means": bool(space.get("include_component_means", True)),
    }
    _validate_constraints(normalized["params"], normalized["constraints"])
    return normalized


def _validate_constraints(params: Dict[str, List[Any]], constraints: List[Dict[str, Any]]) -> None:
    valid_types = {
        "sum_eq",
        "sum_lte",
        "sum_gte",
        "sum_abs_eq",
        "sum_abs_lte",
        "sum_abs_gte",
    }
    for idx, raw in enumerate(constraints):
        if not isinstance(raw, dict):
            raise AblationError(f"Constraint #{idx+1} must be an object.")
        ctype = str(raw.get("type", "")).strip()
        if ctype not in valid_types:
            raise AblationError(f"Constraint #{idx+1} has unsupported type '{ctype}'.")
        cparams = raw.get("params")
        if not isinstance(cparams, list) or not cparams:
            raise AblationError(f"Constraint #{idx+1} must include non-empty 'params'.")
        unknown = [p for p in cparams if p not in params]
        if unknown:
            raise AblationError(f"Constraint #{idx+1} references unknown params: {unknown}")
        if "value" not in raw:
            raise AblationError(f"Constraint #{idx+1} must include 'value'.")
        _safe_float(raw.get("value"))


def _constraint_holds(combo: Dict[str, Any], constraint: Dict[str, Any]) -> bool:
    ctype = str(constraint["type"])
    names = list(constraint["params"])
    vals = [_safe_float(combo.get(name), float("nan")) for name in names]
    if any(math.isnan(v) for v in vals):
        raise AblationError(f"Constraint {constraint} requires numeric values.")
    total = sum(vals)
    if "sum_abs_" in ctype:
        total = sum(abs(v) for v in vals)
    target = float(constraint["value"])
    tol = float(constraint.get("tol", 1e-9))
    if ctype.endswith("_eq"):
        return abs(total - target) <= tol
    if ctype.endswith("_lte"):
        return total <= target + tol
    if ctype.endswith("_gte"):
        return total + tol >= target
    raise AblationError(f"Unhandled constraint type: {ctype}")


def _iter_grid_configs(
    search_space: Dict[str, Any],
    *,
    max_configs: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    params = search_space["params"]
    keys = sorted(params.keys())
    constraints = list(search_space.get("constraints", []))
    produced = 0
    for values in itertools.product(*(params[k] for k in keys)):
        combo = {k: v for k, v in zip(keys, values)}
        if constraints and not all(_constraint_holds(combo, c) for c in constraints):
            continue
        canonical = json.dumps(combo, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        cfg_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
        produced += 1
        yield {
            "config_index": produced,
            "config_id": f"cfg_{cfg_hash[:12]}",
            "config_hash": cfg_hash,
            "params": combo,
        }
        if max_configs is not None and produced >= int(max_configs):
            break


def _raw_product_count(search_space: Dict[str, Any]) -> int:
    total = 1
    for vals in search_space["params"].values():
        total *= len(vals)
    return total


def _search_space_hash(search_space: Dict[str, Any]) -> str:
    canonical = json.dumps(search_space, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _config_fieldnames(search_space: Dict[str, Any]) -> List[str]:
    cols = ["config_index", "config_id", "config_hash"]
    for pname in sorted(search_space["params"].keys()):
        cols.append(f"param__{_safe_col(pname)}")
    cols.append("params_json")
    return cols


def _config_row(cfg: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "config_index": cfg["config_index"],
        "config_id": cfg["config_id"],
        "config_hash": cfg["config_hash"],
        "params_json": json.dumps(cfg["params"], sort_keys=True, ensure_ascii=False),
    }
    for pname in sorted(search_space["params"].keys()):
        row[f"param__{_safe_col(pname)}"] = cfg["params"].get(pname, "")
    return row


def _trial_component_keys(include_component_means: bool) -> List[str]:
    if not include_component_means:
        return []
    return [
        "bridge_coverage",
        "best_path",
        "redundancy",
        "fragility",
        "coherence",
        "coverage",
    ]


def _paper_result_fieldnames(search_space: Dict[str, Any]) -> List[str]:
    out = [
        "config_id",
        "paper_key",
        "paper_id",
        "sheet",
        "title",
        "rating",
        "n_rows",
        "n_success",
        "graph_score_mean",
        "graph_score_std",
        "graph_score_min",
        "graph_score_max",
        "score_sum",
        "score_sumsq",
    ]
    for comp in _trial_component_keys(search_space.get("include_component_means", True)):
        out.append(f"component__{comp}__mean")
    return out


def _apply_node_metrics_fast(kg, node_results: Dict[str, Dict[str, Any]]) -> None:
    for node_id, result in (node_results or {}).items():
        nid = str(node_id)
        if nid not in kg.nodes:
            continue
        if not isinstance(result, dict):
            continue
        vals = {m: _clip_metric(_safe_float(result.get(m), 0.5)) for m in METRICS}
        kg.update_node_metrics(node_id=nid, **vals)


def _clip_metric(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _apply_numeric_overrides(
    kg,
    overrides: Dict[str, Any],
    *,
    normalize_metric_weights: bool = False,
) -> None:
    metric_updates: Dict[str, float] = {}
    node_q_updates: Dict[str, Dict[str, float]] = {}
    edge_updates: Dict[str, float] = {}
    graph_updates: Dict[str, float] = {}
    penalty_updates: Dict[str, Any] = {}

    for key, raw in overrides.items():
        parts = str(key).split(".")
        if len(parts) < 2:
            raise AblationError(f"Unsupported grid-search param '{key}'.")
        head = parts[0]
        if head == "metric_w" and len(parts) == 2:
            metric = parts[1]
            if metric not in METRICS:
                raise AblationError(f"Unknown metric_w field '{metric}'.")
            val = float(raw)
            if val < 0.0:
                raise AblationError(f"metric_w values must be non-negative: {key}={raw}")
            metric_updates[metric] = val
            continue
        if head == "node_q" and len(parts) == 3:
            role, metric = parts[1], parts[2]
            if metric not in METRICS:
                raise AblationError(f"Unknown node_q metric '{metric}' in '{key}'.")
            node_q_updates.setdefault(role, {})[metric] = float(raw)
            continue
        if head == "edge_w" and len(parts) == 2:
            if not hasattr(kg.edge_w, parts[1]):
                raise AblationError(f"Unknown edge_w field '{parts[1]}'.")
            edge_updates[parts[1]] = float(raw)
            continue
        if head == "graph_w" and len(parts) == 2:
            if not hasattr(kg.graph_w, parts[1]):
                raise AblationError(f"Unknown graph_w field '{parts[1]}'.")
            graph_updates[parts[1]] = float(raw)
            continue
        if head == "penalty" and len(parts) == 2:
            if not hasattr(kg.penalty, parts[1]):
                raise AblationError(f"Unknown penalty field '{parts[1]}'.")
            penalty_updates[parts[1]] = raw
            continue
        raise AblationError(f"Unsupported grid-search param '{key}'.")

    needs_recompute = False

    if metric_updates:
        curr = dict(kg.metric_w.weights)
        curr.update(metric_updates)
        if normalize_metric_weights and curr:
            avg = sum(curr.values()) / len(curr)
            avg = avg if avg > 0 else 1.0
            curr = {k: (v / avg) for k, v in curr.items()}
        kg.metric_w.weights = curr
        kg._global_metric_weight_version += 1
        kg._weighted_metrics_cache.clear()
        kg._node_quality_cache.clear()
        needs_recompute = True

    if node_q_updates:
        per_role = {str(k): dict(v) for k, v in dict(kg.node_q.per_role).items()}
        for role, metric_map in node_q_updates.items():
            merged = dict(per_role.get(role, {}))
            merged.update(metric_map)
            per_role[role] = merged
        kg.node_q.per_role = per_role
        kg._node_quality_cache.clear()
        needs_recompute = True

    if edge_updates:
        for key, value in edge_updates.items():
            setattr(kg.edge_w, key, float(value))
        needs_recompute = True

    if penalty_updates:
        for key, value in penalty_updates.items():
            current = getattr(kg.penalty, key)
            if isinstance(current, bool):
                setattr(kg.penalty, key, bool(value))
            elif isinstance(current, str):
                setattr(kg.penalty, key, str(value))
            else:
                setattr(kg.penalty, key, float(value))
        needs_recompute = True

    if graph_updates:
        for key, value in graph_updates.items():
            setattr(kg.graph_w, key, float(value))

    if needs_recompute:
        kg.recompute_all_confidences()


def _score_trial_for_config(
    *,
    kgr,
    dag_json: Dict[str, Any],
    node_results: Dict[str, Dict[str, Any]],
    config_params: Dict[str, Any],
    reconcile: str,
    normalize_metric_weights: bool,
) -> Dict[str, Any]:
    kg, rep = _build_kg_from_dag(kgr, copy.deepcopy(dag_json), reconcile=reconcile)
    if rep.get("errors"):
        return {"success": 0, "graph_score": float("nan")}
    _apply_node_metrics_fast(kg, node_results)
    _apply_numeric_overrides(
        kg,
        config_params,
        normalize_metric_weights=normalize_metric_weights,
    )
    score, details = kg.graph_score()
    row = {
        "success": 1,
        "graph_score": _safe_float(score),
    }
    for key, value in (details or {}).items():
        row[f"component__{key}"] = _safe_float(value)
    return row


def _summarize_trial_rows_compact(
    rows: Iterable[Dict[str, Any]],
    *,
    include_component_means: bool,
) -> Dict[str, Any]:
    n_rows = 0
    n_success = 0
    score_sum = 0.0
    score_sumsq = 0.0
    score_min = float("nan")
    score_max = float("nan")
    comp_sum: Dict[str, float] = {}
    comp_n: Dict[str, int] = {}

    for row in rows:
        n_rows += 1
        if str(row.get("success")) not in {"1", "True", "true"}:
            continue
        score = _safe_float(row.get("graph_score"), float("nan"))
        if math.isnan(score):
            continue
        n_success += 1
        score_sum += score
        score_sumsq += score * score
        score_min = score if math.isnan(score_min) else min(score_min, score)
        score_max = score if math.isnan(score_max) else max(score_max, score)
        if include_component_means:
            for key, value in row.items():
                if not key.startswith("component__"):
                    continue
                v = _safe_float(value, float("nan"))
                if math.isnan(v):
                    continue
                comp_sum[key] = comp_sum.get(key, 0.0) + v
                comp_n[key] = comp_n.get(key, 0) + 1

    mean = (score_sum / n_success) if n_success else float("nan")
    var = (score_sumsq / n_success - mean * mean) if n_success else float("nan")
    if not math.isnan(var) and var < 0.0:
        var = 0.0
    out = {
        "n_rows": n_rows,
        "n_success": n_success,
        "graph_score_mean": mean,
        "graph_score_std": math.sqrt(var) if not math.isnan(var) else float("nan"),
        "graph_score_min": score_min,
        "graph_score_max": score_max,
        "score_sum": score_sum,
        "score_sumsq": score_sumsq,
    }
    if include_component_means:
        for key in sorted(comp_sum):
            out[f"{key}__mean"] = comp_sum[key] / max(1, comp_n.get(key, 0))
    return out


def _worker_score_paper(
    *,
    paper: Dict[str, Any],
    search_space: Dict[str, Any],
    reconcile: str,
    max_configs: Optional[int],
) -> List[Dict[str, Any]]:
    kgr = _import_scorer()
    trial_inputs: List[Tuple[int, int, Dict[str, Any], Dict[str, Dict[str, Any]]]] = []
    for k, m, dag_path, ns_path in paper["trial_index"]:
        dag_json, node_results = _load_trial_inputs(dag_path, ns_path)
        trial_inputs.append((int(k), int(m), dag_json, node_results))

    include_component_means = bool(search_space.get("include_component_means", True))
    normalize_metric_weights = bool(search_space.get("normalize_metric_weights", False))

    out_rows: List[Dict[str, Any]] = []
    for cfg in _iter_grid_configs(search_space, max_configs=max_configs):
        rows = []
        for _, _, dag_json, node_results in trial_inputs:
            try:
                r = _score_trial_for_config(
                    kgr=kgr,
                    dag_json=dag_json,
                    node_results=_node_results_copy(node_results),
                    config_params=cfg["params"],
                    reconcile=reconcile,
                    normalize_metric_weights=normalize_metric_weights,
                )
            except Exception:
                r = {"success": 0, "graph_score": float("nan")}
            rows.append(r)
        summary = _summarize_trial_rows_compact(rows, include_component_means=include_component_means)
        summary.update(
            {
                "config_id": cfg["config_id"],
                "paper_key": paper["paper_key"],
                "paper_id": paper["paper_id"],
                "sheet": paper["sheet"],
                "title": paper["title"],
                "rating": paper["rating"],
            }
        )
        out_rows.append(summary)
    return out_rows


def _aggregate_config_rows(
    per_paper_rows: Iterable[Dict[str, Any]],
    *,
    config_index: Dict[str, Dict[str, Any]],
    search_space: Dict[str, Any],
) -> List[Dict[str, Any]]:
    include_component_means = bool(search_space.get("include_component_means", True))
    aggs: Dict[str, Dict[str, Any]] = {}

    for row in per_paper_rows:
        cfg_id = str(row["config_id"])
        agg = aggs.setdefault(
            cfg_id,
            {
                "config_id": cfg_id,
                "n_papers": 0,
                "n_rows": 0,
                "n_success": 0,
                "score_sum": 0.0,
                "score_sumsq": 0.0,
                "graph_score_min": float("nan"),
                "graph_score_max": float("nan"),
                "paper_means": [],
                "paper_ratings": [],
                "component_weighted_sum": {},
                "component_weighted_n": {},
            },
        )
        agg["n_papers"] += 1
        agg["n_rows"] += int(row.get("n_rows", 0) or 0)
        agg["n_success"] += int(row.get("n_success", 0) or 0)
        agg["score_sum"] += _safe_float(row.get("score_sum"), 0.0)
        agg["score_sumsq"] += _safe_float(row.get("score_sumsq"), 0.0)
        rmin = _safe_float(row.get("graph_score_min"), float("nan"))
        rmax = _safe_float(row.get("graph_score_max"), float("nan"))
        if not math.isnan(rmin):
            agg["graph_score_min"] = rmin if math.isnan(agg["graph_score_min"]) else min(agg["graph_score_min"], rmin)
        if not math.isnan(rmax):
            agg["graph_score_max"] = rmax if math.isnan(agg["graph_score_max"]) else max(agg["graph_score_max"], rmax)
        mean = _safe_float(row.get("graph_score_mean"), float("nan"))
        if not math.isnan(mean):
            agg["paper_means"].append(mean)
            agg["paper_ratings"].append(str(row.get("rating", "")))
        if include_component_means:
            n_success = int(row.get("n_success", 0) or 0)
            if n_success > 0:
                for key, value in row.items():
                    if not str(key).startswith("component__") or not str(key).endswith("__mean"):
                        continue
                    v = _safe_float(value, float("nan"))
                    if math.isnan(v):
                        continue
                    agg["component_weighted_sum"][key] = agg["component_weighted_sum"].get(key, 0.0) + v * n_success
                    agg["component_weighted_n"][key] = agg["component_weighted_n"].get(key, 0) + n_success

    summary_rows: List[Dict[str, Any]] = []
    for cfg_id in sorted(aggs.keys(), key=lambda cid: int(config_index[cid]["config_index"])):
        agg = aggs[cfg_id]
        n_success = int(agg["n_success"])
        mean = (agg["score_sum"] / n_success) if n_success else float("nan")
        var = (agg["score_sumsq"] / n_success - mean * mean) if n_success else float("nan")
        if not math.isnan(var) and var < 0.0:
            var = 0.0
        row = {
            "config_id": cfg_id,
            "config_index": config_index[cfg_id]["config_index"],
            "config_hash": config_index[cfg_id]["config_hash"],
            "n_papers": agg["n_papers"],
            "n_rows": agg["n_rows"],
            "n_success": n_success,
            "graph_score_mean": mean,
            "graph_score_std": math.sqrt(var) if not math.isnan(var) else float("nan"),
            "graph_score_min": agg["graph_score_min"],
            "graph_score_max": agg["graph_score_max"],
        }

        paper_means = list(agg["paper_means"])
        paper_ratings = list(agg["paper_ratings"])
        rating_ord_map = {"bad": 0, "neutral": 1, "good": 2}
        xs_ord: List[float] = []
        ys: List[float] = []
        auc_labels: List[int] = []
        auc_scores: List[float] = []
        for rating, score in zip(paper_ratings, paper_means):
            r = str(rating or "").strip().lower()
            if r in rating_ord_map:
                xs_ord.append(float(rating_ord_map[r]))
                ys.append(float(score))
            if r in {"good", "bad"}:
                auc_labels.append(1 if r == "good" else 0)
                auc_scores.append(float(score))
        row["spearman_good_neutral_bad"] = _spearman(xs_ord, ys) if len(xs_ord) >= 2 else float("nan")
        row["auc_good_vs_bad"] = (
            _auc_good_bad(auc_labels, auc_scores)
            if len(auc_labels) >= 2 and len(set(auc_labels)) == 2
            else float("nan")
        )

        if include_component_means:
            for key in sorted(agg["component_weighted_sum"].keys()):
                denom = max(1, int(agg["component_weighted_n"].get(key, 0)))
                row[key] = agg["component_weighted_sum"][key] / denom

        cfg_meta = config_index[cfg_id]
        for pname, value in cfg_meta["params"].items():
            row[f"param__{_safe_col(pname)}"] = value
        row["params_json"] = json.dumps(cfg_meta["params"], sort_keys=True, ensure_ascii=False)
        summary_rows.append(row)
    return summary_rows


def _write_config_index(
    *,
    search_dir: Path,
    search_space: Dict[str, Any],
    max_configs: Optional[int],
    reuse_cache: bool,
) -> Dict[str, Dict[str, Any]]:
    path = search_dir / "config_index.csv"
    if reuse_cache and path.exists():
        rows = _csv_read_rows(path)
        out = {}
        for row in rows:
            cfg_id = str(row["config_id"])
            params = json.loads(row["params_json"])
            out[cfg_id] = {
                "config_index": int(row["config_index"]),
                "config_id": cfg_id,
                "config_hash": str(row["config_hash"]),
                "params": params,
            }
        return out

    rows: List[Dict[str, Any]] = []
    out: Dict[str, Dict[str, Any]] = {}
    for cfg in _iter_grid_configs(search_space, max_configs=max_configs):
        rows.append(_config_row(cfg, search_space))
        out[cfg["config_id"]] = cfg
    if rows:
        _csv_write(path, rows, _config_fieldnames(search_space))
    return out


def _load_cached_paper_result(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return _csv_read_rows(path)


def run_numeric_grid_search(
    *,
    runs_root: str,
    out_root: str,
    search_space_path: str,
    paper_ids: Optional[set[str]] = None,
    num_shards: int = 1,
    shard_index: int = 0,
    max_papers: Optional[int] = None,
    max_configs: Optional[int] = None,
    reconcile: str = "prefer_parents",
    max_workers: Optional[int] = None,
    parallel_backend: str = "auto",
    reuse_cache: bool = True,
    force: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    runs_root_p = Path(runs_root)
    out_root_p = Path(out_root)
    search_space = _load_search_space(search_space_path)
    search_hash = _search_space_hash(search_space)
    search_name = search_space["name"]
    search_dir = out_root_p / search_name
    search_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = search_dir / "search_manifest.json"
    if reuse_cache and manifest_path.exists():
        manifest = _json_load(manifest_path)
        prev_hash = str(manifest.get("search_space_hash", ""))
        if prev_hash and prev_hash != search_hash and not force:
            raise AblationError(
                f"Output directory {search_dir} already contains a different search space hash. "
                "Use --force or a different --out-root."
            )
        prev_max_configs = manifest.get("max_configs", None)
        if prev_max_configs != max_configs and not force:
            raise AblationError(
                f"Output directory {search_dir} was created with max_configs={prev_max_configs}, "
                f"but this run requested max_configs={max_configs}. Use --force or a different --out-root."
            )

    all_papers = _discover_papers(runs_root_p)
    papers = _filter_papers(
        all_papers,
        paper_ids=paper_ids,
        num_shards=int(num_shards),
        shard_index=int(shard_index),
        max_papers=max_papers,
    )

    config_index = _write_config_index(
        search_dir=search_dir,
        search_space=search_space,
        max_configs=max_configs,
        reuse_cache=reuse_cache and not force,
    )
    config_count = len(config_index)
    raw_product = _raw_product_count(search_space)

    workers = int(max_workers or 0)
    if workers <= 0:
        workers = max(1, min(len(papers) or 1, (os.cpu_count() or 1)))

    manifest = {
        "search_name": search_name,
        "description": search_space.get("description", ""),
        "runs_root": str(runs_root_p.resolve()),
        "search_space_path": str(Path(search_space_path).resolve()),
        "search_space_hash": search_hash,
        "raw_product_count": raw_product,
        "config_count": config_count,
        "num_papers_discovered": len(all_papers),
        "num_papers_selected": len(papers),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "max_configs": max_configs,
        "reconcile": reconcile,
        "max_workers": workers,
        "parallel_backend": parallel_backend,
        "normalize_metric_weights": bool(search_space.get("normalize_metric_weights", False)),
        "include_component_means": bool(search_space.get("include_component_means", True)),
        "search_space": search_space,
    }
    _json_dump(manifest_path, manifest)

    if dry_run:
        return {
            "search_name": search_name,
            "config_count": config_count,
            "raw_product_count": raw_product,
            "num_papers_selected": len(papers),
            "out_dir": str(search_dir),
        }

    paper_results_dir = search_dir / "paper_results"
    paper_results_dir.mkdir(parents=True, exist_ok=True)
    per_paper_rows: List[Dict[str, Any]] = []

    pending: List[Dict[str, Any]] = []
    for paper in papers:
        cache_path = paper_results_dir / f"{paper['paper_key']}.csv"
        if reuse_cache and not force and cache_path.exists():
            per_paper_rows.extend(_load_cached_paper_result(cache_path))
        else:
            pending.append(paper)

    if pending and workers == 1:
        for idx, paper in enumerate(pending, 1):
            rows = _worker_score_paper(
                paper=paper,
                search_space=search_space,
                reconcile=reconcile,
                max_configs=max_configs,
            )
            cache_path = paper_results_dir / f"{paper['paper_key']}.csv"
            if rows:
                _csv_write(cache_path, rows, _paper_result_fieldnames(search_space))
                per_paper_rows.extend(rows)
            if verbose:
                print(f"[grid] completed paper {idx}/{len(pending)}: {paper['paper_key']}")
    elif pending:
        def _run_with_executor(executor_cls):
            with executor_cls(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        _worker_score_paper,
                        paper=paper,
                        search_space=search_space,
                        reconcile=reconcile,
                        max_configs=max_configs,
                    ): paper
                    for paper in pending
                }
                done = 0
                for fut in as_completed(futs):
                    paper = futs[fut]
                    rows = fut.result()
                    cache_path = paper_results_dir / f"{paper['paper_key']}.csv"
                    if rows:
                        _csv_write(cache_path, rows, _paper_result_fieldnames(search_space))
                        per_paper_rows.extend(rows)
                    done += 1
                    if verbose:
                        print(f"[grid] completed paper {done}/{len(pending)}: {paper['paper_key']}")

        backend = str(parallel_backend or "auto").strip().lower()
        if backend not in {"auto", "process", "thread"}:
            raise AblationError(f"Unknown parallel_backend='{parallel_backend}'")
        if backend == "thread":
            _run_with_executor(ThreadPoolExecutor)
        elif backend == "process":
            _run_with_executor(ProcessPoolExecutor)
        else:
            try:
                _run_with_executor(ProcessPoolExecutor)
            except PermissionError:
                if verbose:
                    print("[grid] process pool unavailable; falling back to thread pool")
                _run_with_executor(ThreadPoolExecutor)

    if per_paper_rows:
        per_paper_fields = _paper_result_fieldnames(search_space)
        _csv_write(search_dir / "per_paper_summary.csv", per_paper_rows, per_paper_fields)

    config_summary_rows = _aggregate_config_rows(
        per_paper_rows,
        config_index=config_index,
        search_space=search_space,
    )
    if config_summary_rows:
        config_fields = sorted({k for row in config_summary_rows for k in row.keys()})
        preferred = [
            "config_index",
            "config_id",
            "config_hash",
            "n_papers",
            "n_rows",
            "n_success",
            "graph_score_mean",
            "graph_score_std",
            "graph_score_min",
            "graph_score_max",
            "spearman_good_neutral_bad",
            "auc_good_vs_bad",
            "params_json",
        ]
        ordered = preferred + [f for f in config_fields if f not in preferred]
        _csv_write(search_dir / "config_summary.csv", config_summary_rows, ordered)

        by_auc = sorted(
            config_summary_rows,
            key=lambda r: _safe_float(r.get("auc_good_vs_bad"), float("-inf")),
            reverse=True,
        )
        by_sp = sorted(
            config_summary_rows,
            key=lambda r: _safe_float(r.get("spearman_good_neutral_bad"), float("-inf")),
            reverse=True,
        )
        _json_dump(
            search_dir / "top_configs.json",
            {
                "top_by_auc_good_vs_bad": by_auc[: min(25, len(by_auc))],
                "top_by_spearman_good_neutral_bad": by_sp[: min(25, len(by_sp))],
            },
        )

    return {
        "search_name": search_name,
        "config_count": config_count,
        "raw_product_count": raw_product,
        "num_papers_selected": len(papers),
        "num_pending_computed": len(pending),
        "max_workers": workers,
        "out_dir": str(search_dir),
    }


def supported_parameter_patterns() -> Dict[str, str]:
    return dict(SUPPORTED_PARAM_PATTERNS)
